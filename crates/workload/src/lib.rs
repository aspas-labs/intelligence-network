use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use thiserror::Error;

const MAX_STRING_LEN: usize = 64 * 1024;
const MAX_LIST_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadKind {
    Training,
    Inference,
    Evaluation,
    Embedding,
    DataPreparation,
    CheckpointConversion,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadManifest {
    pub workload_id: String,
    pub name: String,
    pub version: String,
    pub kind: WorkloadKind,
    pub owner: String,
    pub model_id: String,
    pub framework: String,
    pub adapter: String,
    pub requirements: Vec<String>,
    pub artifacts: Vec<String>,
    pub datasets: Vec<String>,
    pub execution_policy: String,
    pub parallelism: String,
    pub checkpoint_policy: String,
    pub fault_policy: String,
    pub security_policy: String,
    pub metadata: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingWorkloadManifest {
    pub workload_id: String,
    pub name: String,
    pub version: String,
    pub kind: WorkloadKind,
    pub owner: String,
    pub model_id: String,
    pub framework: String,
    pub adapter: String,
    pub requirements: Vec<String>,
    pub artifacts: Vec<String>,
    pub datasets: Vec<String>,
    pub execution_policy: String,
    pub parallelism: String,
    pub checkpoint_policy: String,
    pub fault_policy: String,
    pub security_policy: String,
    pub metadata: Value,
    pub model_identity: String,
    pub architecture_name: String,
    pub parameter_count: u64,
    pub parameter_bytes: u64,
    pub precision: String,
    pub optimizer: String,
    pub optimizer_state_bytes: u64,
    pub activation_estimate: u64,
    pub tokenizer_artifact: String,
    pub dataset_artifacts: Vec<String>,
    pub data_locality: String,
    pub checkpoint_interval: u64,
    pub minimum_workers: u16,
    pub maximum_workers: u16,
    pub minimum_ram: u64,
    pub minimum_gpu_memory: u64,
    pub preferred_backends: Vec<String>,
    pub required_kernels: Vec<String>,
    pub parallelism_options: Vec<String>,
    pub maximum_staleness: u16,
    pub local_steps: u16,
    pub replication_factor: u16,
    pub fault_tolerance: String,
    pub adapter_name: String,
    pub adapter_version: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkloadError {
    #[error("field {field} is empty")]
    Empty { field: &'static str },
    #[error("field {field} exceeds maximum length {max}")]
    TooLong { field: &'static str, max: usize },
    #[error("field {field} exceeds maximum count {max}")]
    TooMany { field: &'static str, max: usize },
    #[error("field {field} is invalid: {reason}")]
    Invalid { field: &'static str, reason: String },
}

fn bounded_text(value: &str, field: &'static str) -> Result<(), WorkloadError> {
    if value.trim().is_empty() {
        return Err(WorkloadError::Empty { field });
    }
    if value.len() > MAX_STRING_LEN {
        return Err(WorkloadError::TooLong {
            field,
            max: MAX_STRING_LEN,
        });
    }
    Ok(())
}

fn bounded_vec<T>(items: &[T], field: &'static str) -> Result<(), WorkloadError> {
    if items.len() > MAX_LIST_LEN {
        return Err(WorkloadError::TooMany {
            field,
            max: MAX_LIST_LEN,
        });
    }
    Ok(())
}

impl WorkloadManifest {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        bounded_text(&self.workload_id, "workload_id")?;
        bounded_text(&self.name, "name")?;
        bounded_text(&self.version, "version")?;
        bounded_text(&self.owner, "owner")?;
        bounded_text(&self.model_id, "model_id")?;
        bounded_text(&self.framework, "framework")?;
        bounded_text(&self.adapter, "adapter")?;
        bounded_vec(&self.requirements, "requirements")?;
        bounded_vec(&self.artifacts, "artifacts")?;
        bounded_vec(&self.datasets, "datasets")?;
        bounded_text(&self.execution_policy, "execution_policy")?;
        bounded_text(&self.parallelism, "parallelism")?;
        bounded_text(&self.checkpoint_policy, "checkpoint_policy")?;
        bounded_text(&self.fault_policy, "fault_policy")?;
        bounded_text(&self.security_policy, "security_policy")?;
        if self.metadata.is_null() {
            return Err(WorkloadError::Invalid {
                field: "metadata",
                reason: "metadata must not be null".to_string(),
            });
        }
        Ok(())
    }
}

impl TrainingWorkloadManifest {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        WorkloadManifest {
            workload_id: self.workload_id.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
            kind: self.kind.clone(),
            owner: self.owner.clone(),
            model_id: self.model_id.clone(),
            framework: self.framework.clone(),
            adapter: self.adapter.clone(),
            requirements: self.requirements.clone(),
            artifacts: self.artifacts.clone(),
            datasets: self.datasets.clone(),
            execution_policy: self.execution_policy.clone(),
            parallelism: self.parallelism.clone(),
            checkpoint_policy: self.checkpoint_policy.clone(),
            fault_policy: self.fault_policy.clone(),
            security_policy: self.security_policy.clone(),
            metadata: self.metadata.clone(),
        }
        .validate()?;

        let allowed_parallelism = [
            "data-parallel",
            "local-sgd",
            "tensor-parallel",
            "pipeline-parallel",
            "hybrid",
        ];
        if !allowed_parallelism.contains(&self.parallelism.as_str()) {
            return Err(WorkloadError::Invalid {
                field: "parallelism",
                reason: "parallelism must be one of the supported reference strategies".to_string(),
            });
        }

        bounded_text(&self.model_identity, "model_identity")?;
        bounded_text(&self.architecture_name, "architecture_name")?;
        bounded_text(&self.precision, "precision")?;
        bounded_text(&self.optimizer, "optimizer")?;
        bounded_text(&self.tokenizer_artifact, "tokenizer_artifact")?;
        bounded_text(&self.data_locality, "data_locality")?;
        bounded_text(&self.fault_tolerance, "fault_tolerance")?;
        bounded_text(&self.adapter_name, "adapter_name")?;
        bounded_text(&self.adapter_version, "adapter_version")?;

        if self.parameter_count == 0 || self.parameter_bytes == 0 {
            return Err(WorkloadError::Invalid {
                field: "parameter_count",
                reason: "parameter_count and parameter_bytes must be greater than zero".to_string(),
            });
        }
        if self.checkpoint_interval == 0 {
            return Err(WorkloadError::Invalid {
                field: "checkpoint_interval",
                reason: "checkpoint_interval must be positive".to_string(),
            });
        }
        if self.minimum_workers == 0 || self.maximum_workers < self.minimum_workers {
            return Err(WorkloadError::Invalid {
                field: "minimum_workers",
                reason: "minimum_workers must be positive and not exceed maximum_workers"
                    .to_string(),
            });
        }
        if self.minimum_ram == 0 {
            return Err(WorkloadError::Invalid {
                field: "minimum_ram",
                reason: "minimum_ram must be positive".to_string(),
            });
        }
        bounded_vec(&self.dataset_artifacts, "dataset_artifacts")?;
        bounded_vec(&self.preferred_backends, "preferred_backends")?;
        bounded_vec(&self.required_kernels, "required_kernels")?;
        bounded_vec(&self.parallelism_options, "parallelism_options")?;

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterProtocolVersion {
    V1,
}

impl AdapterProtocolVersion {
    pub fn as_u16(self) -> u16 {
        match self {
            Self::V1 => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdapterMessage {
    Hello {
        adapter_name: String,
        adapter_version: String,
    },
    Probe {
        capability: String,
    },
    Prepare {
        manifest: String,
    },
    LoadModel {
        artifact_id: String,
    },
    LoadShard {
        shard_id: String,
        payload: String,
    },
    LoadOptimizerState {
        artifact_id: String,
    },
    TrainWindow {
        request: String,
    },
    ExportUpdate {
        step: u64,
        generation: u64,
    },
    ImportUpdate {
        update: String,
    },
    Checkpoint {
        generation: u64,
    },
    Restore {
        checkpoint_id: String,
    },
    Health,
    Cancel,
    Shutdown,
}

impl AdapterMessage {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        match self {
            Self::Hello {
                adapter_name,
                adapter_version,
            } => {
                bounded_text(adapter_name, "adapter_name")?;
                bounded_text(adapter_version, "adapter_version")?;
            }
            Self::Probe { capability } => bounded_text(capability, "capability")?,
            Self::Prepare { manifest } => bounded_text(manifest, "manifest")?,
            Self::LoadModel { artifact_id } => bounded_text(artifact_id, "artifact_id")?,
            Self::LoadShard { shard_id, payload } => {
                bounded_text(shard_id, "shard_id")?;
                bounded_text(payload, "payload")?;
            }
            Self::LoadOptimizerState { artifact_id } => bounded_text(artifact_id, "artifact_id")?,
            Self::TrainWindow { request } => bounded_text(request, "request")?,
            Self::ExportUpdate { .. } => {}
            Self::ImportUpdate { update } => bounded_text(update, "update")?,
            Self::Checkpoint { generation } => {
                if *generation == 0 {
                    return Err(WorkloadError::Invalid {
                        field: "generation",
                        reason: "generation must be positive".to_string(),
                    });
                }
            }
            Self::Restore { checkpoint_id } => bounded_text(checkpoint_id, "checkpoint_id")?,
            Self::Health | Self::Cancel | Self::Shutdown => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterEnvelope {
    pub protocol_version: AdapterProtocolVersion,
    pub message: AdapterMessage,
}

impl AdapterEnvelope {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        match self.protocol_version {
            AdapterProtocolVersion::V1 => {}
        }
        self.message.validate()
    }
}

pub fn builtin_training_workload() -> TrainingWorkloadManifest {
    TrainingWorkloadManifest {
        workload_id: "builtin.training.v1".to_string(),
        name: "reference-training".to_string(),
        version: "1.0.0".to_string(),
        kind: WorkloadKind::Training,
        owner: "local.operator".to_string(),
        model_id: "builtin.model.reference".to_string(),
        framework: "reference".to_string(),
        adapter: "training.reference.v1".to_string(),
        requirements: vec!["cpu".to_string(), "bounded-memory".to_string()],
        artifacts: vec!["training.reference.model".to_string()],
        datasets: vec![],
        execution_policy: "local-sgd".to_string(),
        parallelism: "local-sgd".to_string(),
        checkpoint_policy: "interval".to_string(),
        fault_policy: "replicate".to_string(),
        security_policy: "bounded".to_string(),
        metadata: serde_json::json!({
            "reference_only": true,
            "workload_family": "training"
        }),
        model_identity: "builtin.model.reference".to_string(),
        architecture_name: "reference-linear".to_string(),
        parameter_count: 1_024,
        parameter_bytes: 8_192,
        precision: "fp32".to_string(),
        optimizer: "sgd".to_string(),
        optimizer_state_bytes: 8_192,
        activation_estimate: 4_096,
        tokenizer_artifact: "builtin.tokenizer.reference".to_string(),
        dataset_artifacts: vec![],
        data_locality: "SELECTIVE".to_string(),
        checkpoint_interval: 2,
        minimum_workers: 1,
        maximum_workers: 8,
        minimum_ram: 256 * 1024 * 1024,
        minimum_gpu_memory: 0,
        preferred_backends: vec!["cpu".to_string()],
        required_kernels: vec![],
        parallelism_options: vec!["local-sgd".to_string()],
        maximum_staleness: 3,
        local_steps: 2,
        replication_factor: 1,
        fault_tolerance: "soft".to_string(),
        adapter_name: "training.reference".to_string(),
        adapter_version: "1.0.0".to_string(),
    }
}

pub fn builtin_adapter_registry() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "training.reference.v1",
            "transport": "stdio",
            "protocol_version": 1,
            "framework": "reference",
            "description": "built-in reference training adapter"
        }),
        serde_json::json!({
            "name": "training.pytorch.v1",
            "transport": "stdio",
            "protocol_version": 1,
            "framework": "pytorch",
            "description": "external PyTorch training adapter"
        }),
        serde_json::json!({
            "name": "training.jax.v1",
            "transport": "stdio",
            "protocol_version": 1,
            "framework": "jax",
            "description": "external JAX training adapter"
        }),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalAdapterTransport {
    Stdio,
    UnixSocket,
    NamedPipe,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRequest {
    pub request_id: u64,
    pub envelope: AdapterEnvelope,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterResponse {
    pub request_id: u64,
    pub ok: bool,
    pub status: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalAdapterRuntimeConfig {
    pub executable: String,
    pub args: Vec<String>,
    pub transport: LocalAdapterTransport,
    pub timeout_ms: u64,
}

impl LocalAdapterRuntimeConfig {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        bounded_text(&self.executable, "executable")?;
        bounded_vec(&self.args, "args")?;
        if self.timeout_ms == 0 || self.timeout_ms > 300_000 {
            return Err(WorkloadError::Invalid {
                field: "timeout_ms",
                reason: "timeout_ms must be between 1 and 300000".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LocalAdapterRuntime {
    pub config: LocalAdapterRuntimeConfig,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    next_request_id: u64,
}

impl LocalAdapterRuntime {
    pub fn new(config: LocalAdapterRuntimeConfig) -> Result<Self, WorkloadError> {
        config.validate()?;
        Ok(Self {
            config,
            child: None,
            stdin: None,
            stdout: None,
            next_request_id: 1,
        })
    }

    pub fn start(&mut self) -> Result<(), WorkloadError> {
        if self.child.is_some() {
            return Ok(());
        }
        let mut child = Command::new(&self.config.executable)
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| WorkloadError::Invalid {
                field: "executable",
                reason: format!("failed to start adapter process: {error}"),
            })?;
        let stdin = child.stdin.take().ok_or_else(|| WorkloadError::Invalid {
            field: "stdin",
            reason: "adapter process did not expose stdin".to_string(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| WorkloadError::Invalid {
            field: "stdout",
            reason: "adapter process did not expose stdout".to_string(),
        })?;
        self.stdin = Some(stdin);
        self.stdout = Some(BufReader::new(stdout));
        self.child = Some(child);
        Ok(())
    }

    pub fn send(&mut self, message: AdapterMessage) -> Result<AdapterResponse, WorkloadError> {
        let envelope = AdapterEnvelope {
            protocol_version: AdapterProtocolVersion::V1,
            message,
        };
        envelope.validate()?;

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request = AdapterRequest {
            request_id,
            envelope,
        };
        let payload = serde_json::to_string(&request).map_err(|error| WorkloadError::Invalid {
            field: "request",
            reason: format!("failed to encode adapter request: {error}"),
        })?;

        let stdin = self.stdin.as_mut().ok_or_else(|| WorkloadError::Invalid {
            field: "adapter",
            reason: "adapter process has not been started".to_string(),
        })?;
        stdin
            .write_all(payload.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|error| WorkloadError::Invalid {
                field: "adapter",
                reason: format!("failed to write adapter request: {error}"),
            })?;

        let stdout = self.stdout.as_mut().ok_or_else(|| WorkloadError::Invalid {
            field: "adapter",
            reason: "adapter process is missing stdout".to_string(),
        })?;
        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .map_err(|error| WorkloadError::Invalid {
                field: "adapter",
                reason: format!("failed to read adapter response: {error}"),
            })?;

        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(WorkloadError::Invalid {
                field: "adapter",
                reason: "adapter returned an empty response".to_string(),
            });
        }

        let response: AdapterResponse =
            serde_json::from_str(trimmed).map_err(|error| WorkloadError::Invalid {
                field: "response",
                reason: format!("failed to decode adapter response: {error}"),
            })?;

        if response.request_id != request_id {
            return Err(WorkloadError::Invalid {
                field: "response",
                reason: format!(
                    "request_id mismatch: expected {request_id}, got {}",
                    response.request_id
                ),
            });
        }

        Ok(response)
    }

    pub fn execute_training_workload(
        &mut self,
        manifest: &TrainingWorkloadManifest,
        step: u64,
    ) -> Result<serde_json::Value, WorkloadError> {
        self.start()?;

        let hello = self.send(AdapterMessage::Hello {
            adapter_name: manifest.adapter_name.clone(),
            adapter_version: manifest.adapter_version.clone(),
        })?;
        let probe = self.send(AdapterMessage::Probe {
            capability: "training".to_string(),
        })?;
        let prepared_manifest =
            serde_json::to_string(manifest).map_err(|error| WorkloadError::Invalid {
                field: "manifest",
                reason: format!("failed to encode manifest: {error}"),
            })?;
        let prepared = self.send(AdapterMessage::Prepare {
            manifest: prepared_manifest,
        })?;
        let loaded_model = self.send(AdapterMessage::LoadModel {
            artifact_id: manifest.model_identity.clone(),
        })?;
        let train_request = serde_json::json!({
            "step": step,
            "model": manifest.model_identity,
            "adapter": manifest.adapter_name,
            "workload_id": manifest.workload_id,
            "parallelism": manifest.parallelism,
        });
        let train = self.send(AdapterMessage::TrainWindow {
            request: serde_json::to_string(&train_request).map_err(|error| {
                WorkloadError::Invalid {
                    field: "request",
                    reason: format!("failed to encode training request: {error}"),
                }
            })?,
        })?;
        let checkpoint = self.send(AdapterMessage::Checkpoint { generation: 1 })?;

        let parse_payload = |response: &AdapterResponse| {
            serde_json::from_str::<serde_json::Value>(&response.payload)
                .unwrap_or_else(|_| serde_json::json!({ "raw": response.payload.clone() }))
        };

        Ok(serde_json::json!({
            "adapter": manifest.adapter_name,
            "workload_id": manifest.workload_id,
            "step": step,
            "status": {
                "hello": parse_payload(&hello),
                "probe": parse_payload(&probe),
                "prepare": parse_payload(&prepared),
                "load_model": parse_payload(&loaded_model),
                "train": parse_payload(&train),
                "checkpoint": parse_payload(&checkpoint),
            },
            "ok": hello.ok && probe.ok && prepared.ok && loaded_model.ok && train.ok && checkpoint.ok,
        }))
    }

    pub fn shutdown(&mut self) -> Result<(), WorkloadError> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.stdin = None;
        self.stdout = None;
        Ok(())
    }
}

pub trait TrainingAdapter {
    fn probe(&self, capability: &str) -> Result<bool, WorkloadError>;
    fn prepare(&self, manifest: &TrainingWorkloadManifest) -> Result<(), WorkloadError>;
    fn load_model(&self, artifact_id: &str) -> Result<(), WorkloadError>;
    fn train_window(&self, request: &str) -> Result<String, WorkloadError>;
    fn export_update(&self, generation: u64, step: u64) -> Result<String, WorkloadError>;
    fn import_update(&self, update: &str) -> Result<(), WorkloadError>;
    fn checkpoint(&self, generation: u64) -> Result<String, WorkloadError>;
    fn restore(&self, checkpoint_id: &str) -> Result<(), WorkloadError>;
    fn health(&self) -> Result<bool, WorkloadError>;
    fn cancel(&self) -> Result<(), WorkloadError>;
    fn shutdown(&self) -> Result<(), WorkloadError>;
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        AdapterEnvelope, AdapterMessage, AdapterProtocolVersion, LocalAdapterRuntimeConfig,
        LocalAdapterTransport, TrainingWorkloadManifest, WorkloadKind, WorkloadManifest,
    };

    fn example_adapter_path() -> PathBuf {
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_root
            .join("..")
            .join("..")
            .join("examples")
            .join("pytorch_reference_adapter.py")
    }

    #[test]
    fn workload_manifest_validates_bounds() {
        let manifest = WorkloadManifest {
            workload_id: "job-001".to_string(),
            name: "demo".to_string(),
            version: "1.0.0".to_string(),
            kind: WorkloadKind::Training,
            owner: "node-0".to_string(),
            model_id: "model-demo".to_string(),
            framework: "pytorch".to_string(),
            adapter: "training.pytorch.v1".to_string(),
            requirements: vec!["gpu:cuda".to_string()],
            artifacts: vec!["artifact-1".to_string()],
            datasets: vec!["dataset-1".to_string()],
            execution_policy: "local-sgd".to_string(),
            parallelism: "data-parallel".to_string(),
            checkpoint_policy: "interval".to_string(),
            fault_policy: "replicate".to_string(),
            security_policy: "bounded".to_string(),
            metadata: serde_json::json!({"purpose": "demo"}),
        };

        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn training_manifest_rejects_invalid_parallelism() {
        let manifest = TrainingWorkloadManifest {
            workload_id: "job-002".to_string(),
            name: "bad".to_string(),
            version: "1.0.0".to_string(),
            kind: WorkloadKind::Training,
            owner: "node-1".to_string(),
            model_id: "model-bad".to_string(),
            framework: "custom".to_string(),
            adapter: "custom-adapter".to_string(),
            requirements: vec![],
            artifacts: vec![],
            datasets: vec![],
            execution_policy: "local-sgd".to_string(),
            parallelism: "not-a-real-strategy".to_string(),
            checkpoint_policy: "interval".to_string(),
            fault_policy: "replicate".to_string(),
            security_policy: "bounded".to_string(),
            metadata: serde_json::json!({}),
            model_identity: "model-bad".to_string(),
            architecture_name: "demo".to_string(),
            parameter_count: 10,
            parameter_bytes: 80,
            precision: "fp16".to_string(),
            optimizer: "adamw".to_string(),
            optimizer_state_bytes: 64,
            activation_estimate: 1024,
            tokenizer_artifact: "tok".to_string(),
            dataset_artifacts: vec!["ds".to_string()],
            data_locality: "LOCAL_ONLY".to_string(),
            checkpoint_interval: 5,
            minimum_workers: 1,
            maximum_workers: 4,
            minimum_ram: 1024,
            minimum_gpu_memory: 0,
            preferred_backends: vec!["cpu".to_string()],
            required_kernels: vec![],
            parallelism_options: vec!["data-parallel".to_string()],
            maximum_staleness: 3,
            local_steps: 4,
            replication_factor: 1,
            fault_tolerance: "soft".to_string(),
            adapter_name: "demo".to_string(),
            adapter_version: "1.0.0".to_string(),
        };

        assert!(manifest.validate().is_err());
    }

    #[test]
    fn adapter_messages_are_versioned_and_bounded() {
        let envelope = AdapterEnvelope {
            protocol_version: AdapterProtocolVersion::V1,
            message: AdapterMessage::Hello {
                adapter_name: "training.pytorch.v1".to_string(),
                adapter_version: "1.0.0".to_string(),
            },
        };

        assert!(envelope.validate().is_ok());
    }

    #[test]
    fn adapter_runtime_config_is_validated() {
        let config = LocalAdapterRuntimeConfig {
            executable: "/bin/cat".to_string(),
            args: vec!["--help".to_string()],
            transport: LocalAdapterTransport::Stdio,
            timeout_ms: 10_000,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn adapter_request_round_trip_serializes() {
        let request = super::AdapterRequest {
            request_id: 7,
            envelope: AdapterEnvelope {
                protocol_version: AdapterProtocolVersion::V1,
                message: AdapterMessage::Health,
            },
        };
        let encoded = serde_json::to_string(&request).unwrap();
        let decoded: super::AdapterRequest = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.request_id, 7);
        assert!(matches!(decoded.envelope.message, AdapterMessage::Health));
    }

    #[test]
    fn builtins_expose_reference_workload_and_adapters() {
        let workload = super::builtin_training_workload();
        assert!(workload.validate().is_ok());
        let registry = super::builtin_adapter_registry();
        assert!(!registry.is_empty());
    }

    #[test]
    fn adapter_runtime_round_trips_via_stdio() {
        let mut runtime = super::LocalAdapterRuntime::new(super::LocalAdapterRuntimeConfig {
            executable: "python3".to_string(),
            args: vec![
                "-c".to_string(),
                "import json, sys; req=json.loads(sys.stdin.readline()); print(json.dumps({'request_id': req['request_id'], 'ok': True, 'status': 'ok', 'payload': json.dumps({'echo': req['envelope']['message']})}))".to_string(),
            ],
            transport: super::LocalAdapterTransport::Stdio,
            timeout_ms: 10_000,
        })
        .unwrap();

        runtime.start().unwrap();
        let response = runtime
            .send(super::AdapterMessage::Health)
            .expect("real adapter should respond over stdio");

        assert!(response.ok);
        assert_eq!(response.status, "ok");
        assert!(response.payload.contains("echo"));
    }

    #[test]
    fn pytorch_reference_adapter_responds_over_stdio() {
        let script = example_adapter_path();
        let mut runtime = super::LocalAdapterRuntime::new(super::LocalAdapterRuntimeConfig {
            executable: "python3".to_string(),
            args: vec![script.to_string_lossy().to_string()],
            transport: super::LocalAdapterTransport::Stdio,
            timeout_ms: 10_000,
        })
        .unwrap();

        runtime.start().unwrap();
        let response = runtime
            .send(super::AdapterMessage::Health)
            .expect("reference PyTorch adapter should answer health checks");

        assert!(response.ok);
        assert_eq!(response.status, "ok");
        assert!(response.payload.contains("training.pytorch.v1"));
    }

    #[test]
    fn workload_execution_pipeline_runs_through_external_adapter() {
        let script = example_adapter_path();
        let mut runtime = super::LocalAdapterRuntime::new(super::LocalAdapterRuntimeConfig {
            executable: "python3".to_string(),
            args: vec![script.to_string_lossy().to_string()],
            transport: super::LocalAdapterTransport::Stdio,
            timeout_ms: 10_000,
        })
        .unwrap();

        let manifest = super::builtin_training_workload();
        let result = runtime
            .execute_training_workload(&manifest, 1)
            .expect("training workflow should execute through runtime adapter");

        assert_eq!(result["workload_id"], manifest.workload_id);
        assert!(result["ok"].as_bool().unwrap_or(false));
        assert!(result["status"]["hello"].is_object());
        assert!(result["status"]["train"].is_object());
        assert!(result["status"]["checkpoint"].is_object());
    }
}
