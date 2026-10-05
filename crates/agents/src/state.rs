use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentRegistry, AgentTaskGraph};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentState {
    pub registry: AgentRegistry,
    pub task_graphs: BTreeMap<String, AgentTaskGraph>,
    pub swarm_ids: Vec<String>,
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AgentError> {
        serde_json::from_slice(bytes).map_err(|error| AgentError::State(error.to_string()))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, AgentError> {
        serde_json::to_vec(self).map_err(|error| AgentError::State(error.to_string()))
    }

    pub fn push_graph(&mut self, key: String, graph: AgentTaskGraph) {
        self.task_graphs.insert(key, graph);
    }
}
