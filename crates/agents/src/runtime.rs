use std::sync::{Arc, Mutex};

use intelligence_protocol::{JobKind, JobRequest, NodeId, PrivacyPolicy};
use intelligence_runtime::{ExecutorSpec, Runtime, RuntimeConfig};

use crate::{
    AgentError, AgentManifest, AgentModelBackend, AgentRegistry, AgentTask, DecisionContext,
    SystemOneDecisionEngine,
};

#[derive(Clone)]
pub struct AgentRuntime {
    runtime: Arc<Runtime>,
    registry: Arc<Mutex<AgentRegistry>>,
    default_model: String,
    decision_engine: SystemOneDecisionEngine,
    model_backend: Option<Arc<dyn AgentModelBackend>>,
}

impl AgentRuntime {
    pub fn new(config: RuntimeConfig) -> Result<Self, AgentError> {
        Self::new_with_backend(config, None)
    }

    pub fn new_with_backend(
        config: RuntimeConfig,
        model_backend: Option<Arc<dyn AgentModelBackend>>,
    ) -> Result<Self, AgentError> {
        let runtime = Runtime::new(config)
            .map_err(|error| AgentError::Runtime(error.to_string()))
            .map(Arc::new)?;

        Ok(Self {
            runtime,
            registry: Arc::new(Mutex::new(AgentRegistry::new())),
            default_model: "builtin.tiny-sentiment.v1".to_string(),
            decision_engine: SystemOneDecisionEngine::new(),
            model_backend,
        })
    }

    pub fn registry(&self) -> Arc<Mutex<AgentRegistry>> {
        Arc::clone(&self.registry)
    }

    pub fn decide(&self, context: &DecisionContext) -> Result<crate::DecisionOutcome, AgentError> {
        if let Some(backend) = self.model_backend.as_ref() {
            self.decision_engine
                .decide_with_backend(backend.as_ref(), context)
        } else {
            Ok(self.decision_engine.decide(context))
        }
    }

    pub async fn execute_task(
        &self,
        task: &AgentTask,
        manifest: &AgentManifest,
        origin: NodeId,
    ) -> Result<intelligence_runtime::ExecutionOutcome, AgentError> {
        let input = if task.input.is_empty() {
            b"{}".to_vec()
        } else {
            task.input.clone()
        };

        let model = manifest
            .model_requirement
            .clone()
            .unwrap_or_else(|| self.default_model.clone());

        let request = JobRequest {
            job_id: task.task_id,
            origin,
            kind: JobKind::Inference,
            capability: task
                .required_capabilities
                .first()
                .cloned()
                .unwrap_or_else(|| "agent.general".to_string()),
            model: None,
            input,
            deadline_ms: task.deadline_ms,
            max_output_bytes: manifest.max_output_bytes,
            privacy: PrivacyPolicy::default(),
        };

        request
            .validate()
            .map_err(|error| AgentError::InvalidInput(error.to_string()))?;

        let admission = self
            .runtime
            .admit(&request)
            .await
            .map_err(|error| AgentError::Runtime(error.to_string()))?;

        let spec = ExecutorSpec::BuiltinText { model };
        let outcome = self
            .runtime
            .execute(admission, &request, &spec)
            .await
            .map_err(|error| AgentError::Runtime(error.to_string()))?;

        Ok(outcome)
    }
}

impl std::fmt::Debug for AgentRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRuntime")
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}
