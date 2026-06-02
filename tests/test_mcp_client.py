from core.mcp_client import MCPClient, parse_tool_call


def test_parse_tool_call_json_and_key_value():
    name, args = parse_tool_call('get_drug_interaction|{"drug_a":"warfarin","drug_b":"aspirin"}')
    assert name == "get_drug_interaction"
    assert args == {"drug_a": "warfarin", "drug_b": "aspirin"}

    name, args = parse_tool_call("get_drug_interaction|drug_a=warfarin,drug_b=aspirin")
    assert name == "get_drug_interaction"
    assert args["drug_b"] == "aspirin"


def test_mcp_client_calls_drug_tool(tmp_path):
    csv_path = tmp_path / "drugs.csv"
    csv_path.write_text(
        "drug_a,drug_b,severity,description,recommendation\n"
        "warfarin,aspirin,high,增加出血风险,咨询医生\n",
        encoding="utf-8",
    )
    client = MCPClient({"tools": {"drug_interaction": {"enabled": True, "local_csv": str(csv_path)}}})

    result = client.call_tool("get_drug_interaction", {"drug_a": "warfarin", "drug_b": "aspirin"})

    assert "high" in result
    assert "增加出血风险" in result
