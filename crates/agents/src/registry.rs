use std::collections::{BTreeMap, BTreeSet};

use intelligence_protocol::NodeId;
use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentManifest, AgentPresence, AgentRole};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentRegistry {
    by_id: BTreeMap<String, AgentPresence>,
    by_capability: BTreeMap<String, BTreeSet<String>>,
    by_role: BTreeMap<String, BTreeSet<String>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_manifest(
        &mut self,
        manifest: &AgentManifest,
        role: AgentRole,
        node_id: NodeId,
        now_ms: u64,
    ) -> Result<(), AgentError> {
        manifest
            .validate()
            .map_err(|error| AgentError::Manifest(error.to_string()))?;
        manifest
            .verify_signature()
            .map_err(|error| AgentError::Manifest(error.to_string()))?;

        let presence = AgentPresence {
            agent_id: manifest.id.clone(),
            node_id,
            role,
            capabilities: manifest.capabilities.clone(),
            tools: manifest.tools.clone(),
            last_heartbeat_ms: now_ms,
            manifest: manifest.clone(),
            metadata: manifest.metadata.clone(),
        };

        self.by_id.insert(manifest.id.clone(), presence.clone());

        for capability in &presence.capabilities {
            self.by_capability
                .entry(capability.clone())
                .or_default()
                .insert(manifest.id.clone());
        }

        self.by_role
            .entry(role.as_str().to_string())
            .or_default()
            .insert(manifest.id.clone());

        Ok(())
    }

    pub fn unregister(&mut self, agent_id: &str) -> bool {
        let Some(presence) = self.by_id.remove(agent_id) else {
            return false;
        };

        for capability in &presence.capabilities {
            if let Some(ids) = self.by_capability.get_mut(capability) {
                ids.remove(agent_id);
                if ids.is_empty() {
                    self.by_capability.remove(capability);
                }
            }
        }

        let role = presence.role.as_str().to_string();
        if let Some(ids) = self.by_role.get_mut(&role) {
            ids.remove(agent_id);
            if ids.is_empty() {
                self.by_role.remove(&role);
            }
        }

        true
    }

    pub fn list(&self) -> Vec<&AgentPresence> {
        self.by_id.values().collect()
    }

    pub fn find_by_capability(&self, capability: &str) -> Vec<&AgentPresence> {
        let Some(agent_ids) = self.by_capability.get(capability) else {
            return Vec::new();
        };
        agent_ids
            .iter()
            .filter_map(|id| self.by_id.get(id))
            .collect()
    }

    pub fn find_for_requirements(
        &self,
        required_capabilities: &[String],
        role_hint: Option<AgentRole>,
    ) -> Vec<&AgentPresence> {
        let mut matches: Vec<&AgentPresence> = self.list();
        matches.retain(|agent| {
            let matches_role = role_hint.map(|role| agent.role == role).unwrap_or(true);
            let capability_score = required_capabilities
                .iter()
                .filter(|capability| {
                    agent
                        .capabilities
                        .iter()
                        .any(|candidate| candidate == *capability)
                })
                .count();
            matches_role && capability_score == required_capabilities.len()
        });
        matches
    }

    pub fn find_best_match(
        &self,
        required_capabilities: &[String],
        role_hint: Option<AgentRole>,
    ) -> Option<&AgentPresence> {
        let mut best: Option<&AgentPresence> = None;
        let mut best_score = 0usize;

        for candidate in self.list() {
            let matches_role = role_hint.map(|role| candidate.role == role).unwrap_or(true);
            if !matches_role {
                continue;
            }
            let score = required_capabilities
                .iter()
                .filter(|capability| {
                    candidate
                        .capabilities
                        .iter()
                        .any(|value| value == *capability)
                })
                .count();
            if score > best_score {
                best = Some(candidate);
                best_score = score;
            }
        }

        best
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}
