import os
import sqlite3
import pytest
from knowledge_agent import KnowledgeAgent

@pytest.fixture
def setup_db(tmp_path):
    db_path = str(tmp_path / "test_kb.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE merchants (name TEXT, category TEXT, amenities TEXT)")
    cur.execute("INSERT INTO merchants VALUES ('Kaspi Coffee', 'Cafe', 'wifi, takeaway')")
    conn.commit()
    conn.close()
    return db_path

def test_knowledge_agent_contradiction(setup_db, mocker):
    agent = KnowledgeAgent(db_path=setup_db)
    
    # Mock LLM extraction
    mock_extracted = {
        "entity_name": "Kaspi Coffee",
        "claimed_amenity": "swimming pool"
    }
    mocker.patch.object(agent.engine, "extract_json", return_value=mock_extracted)
    
    res = agent.verify_factuality("Kaspi Coffee-де бассейн бар өте ұнады")
    assert res["hallucination_detected"] is True
    assert "swimming pool" in res["reason"]
