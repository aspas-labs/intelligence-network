use intelligence_protocol::{JobId, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use thiserror::Error;

pub const MAX_AGENT_TASKS: usize = 256;
pub const MAX_TASK_DEPENDENCIES: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AgentTaskState {
    Pending,
    Ready,
    Assigned,
    Running,
    Waiting,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentTask {
    pub task_id: JobId,
    pub parent_id: Option<JobId>,
    pub dependencies: Vec<JobId>,
    pub required_capabilities: Vec<String>,
    pub priority: i16,
    pub deadline_ms: u64,
    pub input: Vec<u8>,
    pub state: AgentTaskState,
    pub assigned_agent: Option<String>,
    pub assigned_node: Option<NodeId>,
    pub retry_count: u8,
    pub max_attempts: u8,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentTaskGraph {
    tasks: BTreeMap<JobId, AgentTask>,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum TaskGraphError {
    #[error("task graph exceeds the maximum task count")]
    TooManyTasks,
    #[error("task graph contains duplicate or mismatched task IDs")]
    DuplicateTask,
    #[error("task {0} depends on a missing task")]
    MissingDependency(JobId),
    #[error("task graph contains a dependency cycle")]
    Cycle,
    #[error("task field {0} is invalid")]
    Invalid(&'static str),
}

impl AgentTaskGraph {
    pub fn tasks(&self) -> &BTreeMap<JobId, AgentTask> {
        &self.tasks
    }

    pub fn insert(&mut self, task: AgentTask) -> Result<(), TaskGraphError> {
        let id = task.task_id;
        if self.tasks.contains_key(&id) {
            return Err(TaskGraphError::DuplicateTask);
        }
        if self.tasks.len() >= MAX_AGENT_TASKS {
            return Err(TaskGraphError::TooManyTasks);
        }
        validate_task(&task)?;
        self.tasks.insert(id, task);
        if let Err(error) = self.validate() {
            self.tasks.remove(&id);
            return Err(error);
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), TaskGraphError> {
        if self.tasks.len() > MAX_AGENT_TASKS {
            return Err(TaskGraphError::TooManyTasks);
        }
        for (id, task) in &self.tasks {
            if id != &task.task_id {
                return Err(TaskGraphError::DuplicateTask);
            }
            validate_task(task)?;
            for dependency in &task.dependencies {
                if !self.tasks.contains_key(dependency) {
                    return Err(TaskGraphError::MissingDependency(*dependency));
                }
            }
        }
        let mut complete = HashSet::with_capacity(self.tasks.len());
        for id in self.tasks.keys().copied() {
            let mut visiting = HashSet::new();
            if has_cycle(id, &self.tasks, &mut visiting, &mut complete) {
                return Err(TaskGraphError::Cycle);
            }
        }
        Ok(())
    }
}

fn validate_task(task: &AgentTask) -> Result<(), TaskGraphError> {
    if task.task_id == JobId::default() || task.deadline_ms == 0 || task.max_attempts == 0 {
        return Err(TaskGraphError::Invalid("identity_or_limits"));
    }
    if task.dependencies.len() > MAX_TASK_DEPENDENCIES {
        return Err(TaskGraphError::Invalid("dependencies"));
    }
    let mut dependencies = HashSet::with_capacity(task.dependencies.len());
    if task
        .dependencies
        .iter()
        .any(|dependency| *dependency == task.task_id || !dependencies.insert(*dependency))
    {
        return Err(TaskGraphError::Invalid("dependencies"));
    }
    if task.required_capabilities.len() > 32
        || task
            .required_capabilities
            .iter()
            .any(|value| value.is_empty() || value.len() > 128)
    {
        return Err(TaskGraphError::Invalid("required_capabilities"));
    }
    if task.input.len() > intelligence_protocol::MAX_JOB_INPUT {
        return Err(TaskGraphError::Invalid("input"));
    }
    Ok(())
}

fn has_cycle(
    id: JobId,
    tasks: &BTreeMap<JobId, AgentTask>,
    visiting: &mut HashSet<JobId>,
    complete: &mut HashSet<JobId>,
) -> bool {
    if complete.contains(&id) {
        return false;
    }
    if !visiting.insert(id) {
        return true;
    }
    if let Some(task) = tasks.get(&id) {
        for dependency in &task.dependencies {
            if has_cycle(*dependency, tasks, visiting, complete) {
                return true;
            }
        }
    }
    visiting.remove(&id);
    complete.insert(id);
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: u8, dependencies: Vec<JobId>) -> AgentTask {
        AgentTask {
            task_id: JobId::from_bytes([id; 16]),
            parent_id: None,
            dependencies,
            required_capabilities: vec!["agent.general".to_string()],
            priority: 0,
            deadline_ms: 1000,
            input: Vec::new(),
            state: AgentTaskState::Pending,
            assigned_agent: None,
            assigned_node: None,
            retry_count: 0,
            max_attempts: 3,
        }
    }

    #[test]
    fn accepts_acyclic_graph_and_rejects_missing_dependencies() {
        let root = JobId::from_bytes([1; 16]);
        let leaf = JobId::from_bytes([2; 16]);
        let mut graph = AgentTaskGraph::default();
        graph.insert(task(1, Vec::new())).unwrap();
        graph.insert(task(2, vec![root])).unwrap();
        assert!(graph.validate().is_ok());
        let missing = JobId::from_bytes([9; 16]);
        assert_eq!(
            graph.insert(task(3, vec![leaf, missing])),
            Err(TaskGraphError::MissingDependency(missing))
        );
    }

    #[test]
    fn rejects_dependency_cycles() {
        let first = task(1, vec![JobId::from_bytes([2; 16])]);
        let second = task(2, vec![JobId::from_bytes([1; 16])]);
        let graph = AgentTaskGraph {
            tasks: BTreeMap::from([(first.task_id, first), (second.task_id, second)]),
        };
        assert_eq!(graph.validate(), Err(TaskGraphError::Cycle));
    }
}
