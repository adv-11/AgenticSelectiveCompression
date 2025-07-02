from langgraph import Graph
from .ingest_agent       import IngestAgent
from .retrieval_agent    import RetrievalAgent
from .orchestrator_agent import OrchestratorAgent
from .cleanup_agent      import CleanupAgent

def main():
    graph = Graph()
    graph.add_agent(IngestAgent())
    graph.add_agent(RetrievalAgent())
    graph.add_agent(OrchestratorAgent())
    graph.add_agent(CleanupAgent())
    graph.run()

if __name__ == "__main__":
    main()
