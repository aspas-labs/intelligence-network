use std::{
    collections::BTreeMap,
    process::{Command, Stdio},
};

use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentModelBackend, DecisionContext, DecisionOutcome};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalProcessLLMConfig {
    pub name: String,
    pub program: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub model: String,
    pub timeout_ms: u64,
    pub system_prompt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalProcessLLMBackend {
    pub config: LocalProcessLLMConfig,
    pub healthy: bool,
}

impl Default for LocalProcessLLMBackend {
    fn default() -> Self {
        Self {
            config: LocalProcessLLMConfig {
                name: "local-llm".to_string(),
                program: "python3".to_string(),
                args: vec!["-c".to_string(), "print('llm-ready')".to_string()],
                env: BTreeMap::new(),
                model: "default".to_string(),
                timeout_ms: 10_000,
                system_prompt:
                    "You are a cautious System One decision assistant for distributed tasks."
                        .to_string(),
            },
            healthy: true,
        }
    }
}

impl LocalProcessLLMBackend {
    pub fn new(config: LocalProcessLLMConfig) -> Self {
        Self {
            healthy: !config.program.is_empty(),
            config,
        }
    }

    fn prompt_for(&self, input: &str, context: &DecisionContext) -> String {
        let facts = if context.facts.is_empty() {
            "none".to_string()
        } else {
            context.facts.join("; ")
        };
        let constraints = if context.constraints.is_empty() {
            "none".to_string()
        } else {
            context.constraints.join("; ")
        };
        format!(
            "{}\nGoal: {}\nFacts: {}\nConstraints: {}\nRisk level: {}\nTask: {}\nUser request: {}",
            self.config.system_prompt,
            context.goal,
            facts,
            constraints,
            context.risk_level,
            context
                .task_id
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            input,
        )
    }
}

impl AgentModelBackend for LocalProcessLLMBackend {
    fn name(&self) -> &str {
        self.config.name.as_str()
    }

    fn health(&self) -> bool {
        self.healthy
    }

    fn generate(
        &self,
        prompt: &str,
        context: &DecisionContext,
    ) -> Result<DecisionOutcome, AgentError> {
        if self.config.program.is_empty() {
            return Err(AgentError::Model(
                "local process backend has no program".to_string(),
            ));
        }

        let full_prompt = self.prompt_for(prompt, context);
        let mut child = Command::new(&self.config.program)
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .envs(&self.config.env)
            .spawn()
            .map_err(|error| {
                AgentError::Model(format!(
                    "failed to spawn {}: {}",
                    self.config.program, error
                ))
            })?;

        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin.write_all(full_prompt.as_bytes()).map_err(|error| {
                AgentError::Model(format!("failed to pass prompt to model: {}", error))
            })?;
        }

        let output = match child.wait_with_output() {
            Ok(result) => result,
            Err(error) => {
                return Err(AgentError::Model(format!(
                    "model execution failed for {}: {}",
                    self.config.name, error,
                )));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() && stdout.is_empty() {
            return Err(AgentError::Model(format!(
                "model process reported an error: {}",
                stderr.trim()
            )));
        }

        let decision = if stdout.is_empty() {
            "continue".to_string()
        } else {
            stdout
        };

        Ok(DecisionOutcome {
            decision: decision.trim().to_string(),
            confidence: if context.risk_level >= 7 { 0.69 } else { 0.82 },
            rationale: format!(
                "Local model backend {} evaluated the task using a System One decision prompt.",
                self.config.name
            ),
            evidence: vec![
                format!("model={}", self.config.model),
                format!("program={}", self.config.program),
            ],
            backend: Some(self.name().to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_backend_builds_prompt_and_survives_default_config() {
        let backend = LocalProcessLLMBackend::default();
        let context = DecisionContext {
            goal: "schedule a safe delegation".to_string(),
            facts: vec!["peer discovered".to_string()],
            constraints: vec!["must be bounded".to_string()],
            task_id: Some("task-llm".to_string()),
            deadline_ms: Some(4000),
            risk_level: 2,
        };

        assert_eq!(backend.name(), "local-llm");
        assert!(
            backend
                .prompt_for("decision", &context)
                .contains("schedule a safe delegation")
        );
    }
}
