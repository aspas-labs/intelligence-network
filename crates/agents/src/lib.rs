//! Bounded primitives for peer-discovered, operator-controlled agents.

mod agent;
mod decision;
mod error;
mod llm;
mod manifest;
mod registry;
mod runtime;
mod scheduler;
mod state;
mod task;

pub use agent::{AgentPresence, AgentRole};
pub use decision::{
    AgentModelBackend, DecisionContext, DecisionMode, DecisionOutcome, SystemOneDecisionEngine,
};
pub use error::AgentError;
pub use llm::{LocalProcessLLMBackend, LocalProcessLLMConfig};
pub use manifest::{AgentManifest, ManifestError};
pub use registry::AgentRegistry;
pub use runtime::AgentRuntime;
pub use scheduler::{AgentScheduleDecision, AgentScheduler};
pub use state::AgentState;
pub use task::{AgentTask, AgentTaskGraph, AgentTaskState, TaskGraphError};
