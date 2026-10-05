use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AgentError {
    #[error("agent manifest is invalid: {0}")]
    Manifest(String),
    #[error("agent state is invalid: {0}")]
    State(String),
    #[error("task graph is invalid: {0}")]
    TaskGraph(String),
    #[error("no suitable agent was found for the requested capability or role")]
    NoSuitableAgent,
    #[error("delegation exceeded the configured depth limit")]
    DelegationDepthLimit,
    #[error("task was cancelled")]
    Cancelled,
    #[error("the requested operation is not supported: {0}")]
    Unsupported(String),
    #[error("runtime rejected the work: {0}")]
    Runtime(String),
    #[error("agent input is invalid: {0}")]
    InvalidInput(String),
    #[error("model backend rejected the request: {0}")]
    Model(String),
    #[error("agent record expired or is no longer valid")]
    Expired,
}
