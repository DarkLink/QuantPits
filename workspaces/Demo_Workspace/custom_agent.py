from quantpits.scripts.deep_analysis.base_agent import BaseAgent, AgentFindings, AnalysisContext

class WorkspaceLocalAgent(BaseAgent):
    name = "workspace_local_plugin"
    description = "A clean plugin agent defined within this workspace."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        # Simple custom analysis logic
        findings = [
            self._make_finding(
                severity="info",
                title="Workspace Local Rule Fired",
                detail="Successfully loaded and executed workspace-isolated custom agent."
            )
        ]
        return AgentFindings(self.name, ctx.window_label, findings, [], {})
