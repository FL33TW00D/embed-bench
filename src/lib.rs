extern crate accelerate_src;

use hf_hub::{api::sync::Api, Repo, RepoType};
use ndarray::Axis;
use ort::{
    tensor::{FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, SessionBuilder,
};
use std::{path::PathBuf, sync::Arc};

mod candle;
use candle::{device, normalize_l2, BertModel, Config};
use embd_core::bge::BGEModel;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub struct Ort {
    tokenizer: tokenizers::Tokenizer,
    session: ort::Session,
}

impl Ort {
    pub fn new() -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file("model/tokenizer.json").unwrap();
        let environment = Arc::new(
            Environment::builder()
                .with_name("Encode")
                .with_log_level(LoggingLevel::Warning)
                .with_execution_providers([ExecutionProvider::cpu()])
                .with_telemetry(false)
                .build()?,
        );

        let session = SessionBuilder::new(&environment)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file("model/model.onnx")?;

        Ok(Self { tokenizer, session })
    }

    pub fn embed(&self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let tokenizer_output = self.tokenizer.encode(sequence, true).unwrap();

        let input_ids = tokenizer_output.get_ids();
        let attention_mask = tokenizer_output.get_attention_mask();
        let token_type_ids = tokenizer_output.get_type_ids();
        let length = input_ids.len();
        // trace!("embedding {} tokens {:?}", length, sequence);

        let inputs_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            input_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let attention_mask_array = ndarray::Array::from_shape_vec(
            (1, length),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )?;

        let token_type_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let outputs = self.session.run([
            InputTensor::from_array(inputs_ids_array.into_dyn()),
            InputTensor::from_array(attention_mask_array.into_dyn()),
            InputTensor::from_array(token_type_ids_array.into_dyn()),
        ])?;

        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let sequence_embedding = &*output_tensor.view();
        let pooled = sequence_embedding
            .mean_axis(Axis(1))
            .unwrap()
            .to_owned()
            .as_slice()
            .unwrap()
            .to_vec();
        let norm = pooled.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let pooled = pooled.iter().map(|x| x / norm).collect::<Vec<_>>();
        Ok(pooled)
    }
}

pub struct Ggml {
    model: Box<dyn llm::Model>,
    inference_session: llm::InferenceSession,
}

pub enum Acceleration {
    None,
    Gpu,
}

impl Ggml {
    pub fn new(acc: Acceleration) -> Self {
        let tokenizer_source = llm::TokenizerSource::HuggingFaceTokenizerFile(PathBuf::from(
            "model/updated_tokenizers.json",
        ));
        let model_architecture = llm::ModelArchitecture::Bert;
        let model_path = PathBuf::from("model/ggml-model-q4_0.bin");

        // Load model
        let mut model_params = llm::ModelParameters::default();
        model_params.use_gpu = match acc {
            Acceleration::None => false,
            Acceleration::Gpu => true,
        };
        let model = llm::load_dynamic(
            Some(model_architecture),
            &model_path,
            tokenizer_source,
            model_params,
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| {
            panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
        });

        let session_config = llm::InferenceSessionConfig {
            ..Default::default()
        };
        let inference_session = model.start_session(session_config);

        Self {
            model,
            inference_session,
        }
    }

    pub fn embed(&mut self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let mut output_request = llm::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;

        let query_token_ids = vocab
            .tokenize(sequence, beginning_of_sentence)?
            .iter()
            .map(|(_, tok)| *tok)
            .collect::<Vec<_>>();

        self.model.evaluate(
            &mut self.inference_session,
            &query_token_ids,
            &mut output_request,
        );
        output_request
            .embeddings
            .ok_or(anyhow::anyhow!("failed to embed"))
    }

    pub fn batch_embed(&mut self, sequences: &[&str]) -> anyhow::Result<Vec<f32>> {
        let mut output_request = llm::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };

        let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;

        let query_token_ids = sequences
            .iter()
            .map(|&sequence| {
                vocab
                    .tokenize(&sequence, beginning_of_sentence)
                    .unwrap()
                    .iter()
                    .map(|(_, tok)| *tok)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let query_token_ids: Vec<_> = query_token_ids.iter().map(AsRef::as_ref).collect();
        self.model.batch_evaluate(
            &mut self.inference_session,
            &query_token_ids,
            &mut output_request,
        );
        // let embedding: Vec<Vec<f32>> = output_request.embeddings.unwrap().chunks(384).map(|chunk| chunk.to_vec()).collect();
        Ok(output_request.embeddings.unwrap())
    }
}

pub struct Candle {
    model: BertModel,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
}

impl Candle {
    pub fn new(cpu: bool) -> anyhow::Result<Self> {
        use candle_nn::VarBuilder;

        let device = device(cpu)?;
        let model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let revision = "refs/pr/21".to_string();

        let repo = Repo::with_revision(model, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            (
                api.get("config.json")?,
                api.get("tokenizer.json")?,
                api.get("model.safetensors")?,
            )
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        let weights = unsafe { candle_core::safetensors::MmapedFile::new(weights_filename)? };
        let weights = weights.deserialize()?;
        let vb = VarBuilder::from_safetensors(vec![weights], candle::DTYPE, &device);
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&mut self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let tokenizer = self
            .tokenizer
            .with_padding(None)
            .with_truncation(None)
            .unwrap();
        let tokens = tokenizer
            .encode(sequence, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let token_ids = candle_core::Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = normalize_l2(&embeddings)?;
        Ok(embeddings.squeeze(0)?.to_vec1()?)
    }
}

pub struct Embd {
    model: BGEModel,
    tokenizer: Tokenizer,
}

impl Embd {
    pub async fn new() -> anyhow::Result<Self> {
        let model_path =
            PathBuf::from("/Users/fleetwood/Code/embed-bench/model/bge-small-en-v1.5-f32.bin");
        let tokenizer_path =
            PathBuf::from("/Users/fleetwood/Code/embed-bench/model/bge-tokenizer.json");
        let bge = BGEModel::from_path(model_path).await.unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        Ok(Embd {
            model: bge,
            tokenizer,
        })
    }

    pub async fn batch_embed(&mut self, sequences: &[&str]) -> anyhow::Result<Vec<f32>> {
        let encoded_batch = self
            .tokenizer
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_to_multiple_of: Some(4),
                pad_id: 0,
                ..Default::default()
            }))
            .encode_batch(sequences.to_vec(), true)
            .unwrap();
        let inputs = self.model.prepare_inputs(encoded_batch);

        let mut result = self.model.model.run(inputs).await.unwrap();
        let embeddings_gpu = result.remove(0);

        let embeddings_cpu = embeddings_gpu
            .into_cpu(self.model.manager.handle())
            .await
            .unwrap();
        unsafe { Ok(embeddings_cpu.into_vec()) }
    }
}
