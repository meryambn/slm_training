import pytest
from pathlib import Path
from tools.file_io import FileIOTool
from tools.web_search import WebSearchTool

def test_file_io_read_and_list(tmp_path):
    # Setup dummy directory & file
    sub_dir = tmp_path / "data"
    sub_dir.mkdir()
    sample_file = sub_dir / "test.txt"
    sample_file.write_text("Hello SLM Agent!\nLine 2", encoding="utf-8")

    tool = FileIOTool(base_dir=str(tmp_path))
    
    # Test list_dir
    list_res = tool.execute(action="list_dir", path=str(sub_dir))
    assert list_res.success is True
    assert "test.txt" in list_res.output

    # Test read_file
    read_res = tool.execute(action="read_file", path=str(sample_file))
    assert read_res.success is True
    assert "Hello SLM Agent!" in read_res.output

def test_file_io_csv_header(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("id,accuracy,f1\n1,0.92,0.91\n2,0.88,0.87", encoding="utf-8")

    tool = FileIOTool(base_dir=str(tmp_path))
    res = tool.execute(action="read_csv_header", path=str(csv_file), max_lines=2)
    assert res.success is True
    assert "accuracy" in res.output

def test_file_io_missing_file(tmp_path):
    tool = FileIOTool(base_dir=str(tmp_path))
    res = tool.execute(action="read_file", path="non_existent.txt")
    assert res.success is False
    assert "not found" in res.error

def test_web_search_empty_query():
    tool = WebSearchTool()
    res = tool.execute(query="   ")
    assert res.success is False
    assert "Empty search query" in res.error

def test_web_search_schema():
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "query" in tool.parameters_schema["required"]
