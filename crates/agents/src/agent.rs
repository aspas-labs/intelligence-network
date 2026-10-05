use intelligence_protocol::{MetadataEntry, NodeId};
use serde::{Deserialize, Serialize};

use crate::AgentManifest;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AgentRole {
    Planner,
    Researcher,
    Coder,
    Math,
    Critic,
    Verifier,
    Memory,
    ToolRunner,
    General,
}

impl AgentRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Planner => "planner",
            Self::Researcher => "researcher",
            Self::Coder => "coder",
            Self::Math => "math",
            Self::Critic => "critic",
            Self::Verifier => "verifier",
            Self::Memory => "memory",
            Self::ToolRunner => "tool-runner",
            Self::General => "general",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "planner" => Some(Self::Planner),
            "researcher" => Some(Self::Researcher),
            "coder" => Some(Self::Coder),
            "math" => Some(Self::Math),
            "critic" => Some(Self::Critic),
            "verifier" => Some(Self::Verifier),
            "memory" => Some(Self::Memory),
            "tool-runner" => Some(Self::ToolRunner),
            "general" => Some(Self::General),
            _ => None,
        }
    }
}

impl std::str::FromStr for AgentRole {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_str(value).ok_or(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentPresence {
    pub agent_id: String,
    pub node_id: NodeId,
    pub role: AgentRole,
    pub capabilities: Vec<String>,
    pub tools: Vec<String>,
    pub last_heartbeat_ms: u64,
    pub manifest: AgentManifest,
    pub metadata: Vec<MetadataEntry>,
}

impl AgentPresence {
    pub fn is_expired(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_heartbeat_ms) > ttl_ms
    }
}
