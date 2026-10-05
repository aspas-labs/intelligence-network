use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::AgentError;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum DecisionMode {
    #[default]
    SystemOne,
    Hybrid,
    RuleBased,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecisionContext {
    pub goal: String,
    pub facts: Vec<String>,
    pub constraints: Vec<String>,
    pub task_id: Option<String>,
    pub deadline_ms: Option<u64>,
    pub risk_level: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub decision: String,
    pub confidence: f32,
    pub rationale: String,
    pub evidence: Vec<String>,
    pub backend: Option<String>,
}

impl Default for DecisionOutcome {
    fn default() -> Self {
        Self {
            decision: "safe-default".to_string(),
            confidence: 0.5,
            rationale:
                "No model backend configured; falling back to a conservative rule-based decision."
                    .to_string(),
            evidence: Vec::new(),
            backend: None,
        }
    }
}

pub trait AgentModelBackend: Send + Sync {
    fn name(&self) -> &str;
    fn health(&self) -> bool;
    fn generate(
        &self,
        prompt: &str,
        context: &DecisionContext,
    ) -> Result<DecisionOutcome, AgentError>;
    fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name(),
            "type": "generic",
            "healthy": self.health(),
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SystemOneDecisionEngine {
    pub mode: DecisionMode,
    pub min_confidence: f32,
    pub max_depth: u8,
    pub use_llm_fallback: bool,
}

impl SystemOneDecisionEngine {
    pub fn new() -> Self {
        Self {
            mode: DecisionMode::SystemOne,
            min_confidence: 0.7,
            max_depth: 4,
            use_llm_fallback: true,
        }
    }

    pub fn decide(&self, context: &DecisionContext) -> DecisionOutcome {
        if context.goal.is_empty() {
            return DecisionOutcome {
                decision: "reject".to_string(),
                confidence: 0.0,
                rationale: "Goal is empty; no safe decision can be made.".to_string(),
                evidence: vec!["empty_goal".to_string()],
                backend: None,
            };
        }

        let risk_bias = if context.risk_level >= 7 {
            "defer"
        } else {
            "continue"
        };
        let evidence = vec![
            format!("goal={}", context.goal),
            format!("constraints={}", context.constraints.join(" | ")),
            format!("facts={}", context.facts.join(" | ")),
        ];

        DecisionOutcome {
            decision: risk_bias.to_string(),
            confidence: if context.risk_level >= 7 { 0.62 } else { 0.81 },
            rationale: "System One evaluates the goal, current facts, and risk level before committing to a path.".to_string(),
            evidence,
            backend: None,
        }
    }

    pub fn decide_with_backend(
        &self,
        backend: &dyn AgentModelBackend,
        context: &DecisionContext,
    ) -> Result<DecisionOutcome, AgentError> {
        let mut decision = backend.generate(
            &format!(
                "You are a System One decision engine. Decide the safest next action for the current objective. Return only a compact structured rationale. Goal: {}",
                context.goal,
            ),
            context,
        )?;
        decision.backend = Some(backend.name().to_string());
        if decision.confidence < self.min_confidence && self.use_llm_fallback {
            let fallback = self.decide(context);
            if fallback.confidence >= self.min_confidence {
                return Ok(fallback);
            }
        }
        Ok(decision)
    }
}

impl<T> AgentModelBackend for Arc<T>
where
    T: AgentModelBackend + ?Sized,
{
    fn name(&self) -> &str {
        (**self).name()
    }

    fn health(&self) -> bool {
        (**self).health()
    }

    fn generate(
        &self,
        prompt: &str,
        context: &DecisionContext,
    ) -> Result<DecisionOutcome, AgentError> {
        (**self).generate(prompt, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticBackend;

    impl AgentModelBackend for StaticBackend {
        fn name(&self) -> &str {
            "static-backend"
        }

        fn health(&self) -> bool {
            true
        }

        fn generate(
            &self,
            _prompt: &str,
            context: &DecisionContext,
        ) -> Result<DecisionOutcome, AgentError> {
            Ok(DecisionOutcome {
                decision: "continue".to_string(),
                confidence: 0.9,
                rationale: format!("backend evaluated {}", context.goal),
                evidence: vec!["model".to_string()],
                backend: Some(self.name().to_string()),
            })
        }
    }

    #[test]
    fn system_one_decision_defaults_to_safe_rule_based_choice() {
        let engine = SystemOneDecisionEngine::new();
        let context = DecisionContext {
            goal: "improve reliability".to_string(),
            facts: vec!["stable runtime".to_string()],
            constraints: vec!["low-risk".to_string()],
            task_id: Some("task-1".to_string()),
            deadline_ms: Some(3000),
            risk_level: 3,
        };

        let decision = engine.decide(&context);
        assert_eq!(decision.decision, "continue");
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn model_backend_can_drive_decision() {
        let engine = SystemOneDecisionEngine::new();
        let backend = StaticBackend;
        let context = DecisionContext {
            goal: "delegate coding task".to_string(),
            facts: vec!["there is a suitable coder".to_string()],
            constraints: vec!["must respect deadline".to_string()],
            task_id: Some("task-2".to_string()),
            deadline_ms: Some(5000),
            risk_level: 2,
        };

        let decision = engine.decide_with_backend(&backend, &context).unwrap();
        assert_eq!(decision.decision, "continue");
        assert_eq!(decision.backend, Some("static-backend".to_string()));
    }
}
