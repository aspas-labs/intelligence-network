use intelligence_protocol::NodeId;
use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentRegistry, AgentRole};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentScheduleDecision {
    pub agent_id: String,
    pub node_id: NodeId,
    pub role: AgentRole,
    pub matched_capabilities: Vec<String>,
    pub score: u32,
}

#[derive(Clone, Debug, Default)]
pub struct AgentScheduler {
    pub max_fan_out: usize,
    pub max_depth: usize,
}

impl AgentScheduler {
    pub fn new(max_fan_out: usize, max_depth: usize) -> Self {
        Self {
            max_fan_out: max_fan_out.max(1),
            max_depth: max_depth.max(1),
        }
    }

    pub fn choose_agent(
        &self,
        registry: &AgentRegistry,
        required_capabilities: &[String],
        role_hint: Option<AgentRole>,
    ) -> Result<AgentScheduleDecision, AgentError> {
        let candidate = registry
            .find_best_match(required_capabilities, role_hint)
            .ok_or(AgentError::NoSuitableAgent)?;

        let matched_capabilities: Vec<String> = required_capabilities
            .iter()
            .filter(|capability| {
                candidate
                    .capabilities
                    .iter()
                    .any(|value| value == *capability)
            })
            .cloned()
            .collect();

        let score = matched_capabilities.len() as u32
            + match candidate.role {
                AgentRole::Planner => 5,
                AgentRole::Researcher => 4,
                AgentRole::Coder => 4,
                AgentRole::Verifier => 3,
                _ => 1,
            };

        Ok(AgentScheduleDecision {
            agent_id: candidate.agent_id.clone(),
            node_id: candidate.node_id,
            role: candidate.role,
            matched_capabilities,
            score,
        })
    }

    pub fn find_best_match(
        &self,
        registry: &AgentRegistry,
        required_capabilities: &[String],
        role_hint: Option<AgentRole>,
    ) -> Result<AgentScheduleDecision, AgentError> {
        self.choose_agent(registry, required_capabilities, role_hint)
    }
}
