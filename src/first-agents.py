from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from anthropic import Anthropic

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # loads regulation-rag-v2/.env

# Define state
class AgentState(BaseModel):
    question: str
    rephrased_question: str = ""
    answer: str = ""
    answer_quality: str = ""  # New field to store quality rating of the answer
    quality_score: int = 0  # New field to store numeric quality score (1-5)

# Agent 1: Question Rephraser
def rephraser(state: AgentState) -> AgentState:
    """Rephrase question to be more specific"""
    client = Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Rephrase this question to be clearer: {state.question}"
        }]
    )
    state.rephrased_question = msg.content[0].text
    return state

# Agent 2: Answer Generator
def answerer(state: AgentState) -> AgentState:
    """Generate answer to the rephrased question"""
    client = Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Answer this question: {state.rephrased_question}"
        }]
    )
    state.answer = msg.content[0].text
    return state

def quality_checker(state: AgentState) -> AgentState:
    """Check quality, score it 1-5, improve if needed"""
    client = Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""You are a quality checker.

Question: {state.question}
Answer: {state.answer}

Rate this answer 1-5 and improve if score is below 4.

Reply in this exact format:
SCORE: [number]
IMPROVED_ANSWER: [answer]"""
        }]
    )
    
    response = msg.content[0].text
    for line in response.split('\n'):
        if line.startswith('SCORE:'):
            state.quality_score = int(line.split(':')[1].strip())
        if line.startswith('IMPROVED_ANSWER:'):
            state.answer = line.split(':', 1)[1].strip()
    
    return state
# Build graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("rephraser", rephraser)
graph_builder.add_node("answerer", answerer)
graph_builder.add_node("quality_checker", quality_checker)
graph_builder.add_edge(START, "rephraser")
graph_builder.add_edge("rephraser", "answerer")
graph_builder.add_edge("answerer", "quality_checker")
graph_builder.add_edge("quality_checker", END)
graph = graph_builder.compile()

# Test it
if __name__ == "__main__":
    initial_state = AgentState(question="What is GDPR?")
    result = graph.invoke(initial_state)
    print(f"Original: {result['question']}")
    print(f"Rephrased: {result['rephrased_question']}")
    print(f"Answer: {result['answer']}")
    print(f"Quality Score: {result['quality_score']}")