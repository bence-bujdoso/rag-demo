"""UI tests for RagDemo 2D RAG Visualization Dashboard"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUIDefinitions:
    """Test that UI variables and definitions are correct.
    
    These tests catch NameError and undefined variable issues at import time
    before the UI runs.
    """
    
    def test_node_timings_from_session_state(self):
        """Verify node_timings is accessed from st.session_state properly."""
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'app.py')
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # Check that we use st.session_state.node_timings, not undefined 'nodes'
        assert 'st.session_state.node_timings' in content, "Should use st.session_state.node_timings"
        assert 'enumerate(nodes)' not in content, "Should not use undefined 'nodes' variable"
    
    def test_node_name_field_used(self):
        """Verify we use the correct node['node_name'] field from NodeTiming TypedDict."""
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'app.py')
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # NodeTiming TypedDict has node_name, not 'name'
        assert 'node["node_name"]' in content or "node['node_name']" in content, "Should use node_name field"
    
    def test_score_has_none_handling(self):
        """Verify score values are handled when None (from retrieved documents)."""
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'app.py')
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # Check for None-safe score handling
        has_none_protection = (
            'doc.get("score", 0.0) or 0.0' in content or
            'float(score) if score is not None else 0' in content or
            'score or 0' in content
        )
        assert has_none_protection, "Score should have None handling"
    
    def test_content_has_none_handling(self):
        """Verify content values are handled when None."""
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'app.py')
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # Check for None-safe content handling
        assert 'doc.get("content"' in content and 'or ""' in content, "Content should have None handling"
    
    def test_plotly_color_valid(self):
        """Verify Plotly color values are valid (not None)."""
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui', 'app.py')
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # When using df['Score'] for color, score should be None-safe
        score_handling_context = content.count('doc.get("score", 0.0) or 0.0')
        assert score_handling_context > 0, "Score should be None-safe for Plotly color use"


class TestVisualizationStructure:
    """Test the structure of the visualization dashboard."""
    
    def test_four_tabs_exist(self):
        """Verify all 3 visualization tabs are defined."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            content = f.read()
        
        tabs = [
            "Pipeline Monitoring",
            "Document Analysis", 
            "Performance"
        ]
        for tab in tabs:
            assert tab in content, f"Tab '{tab}' should exist in UI"
    
    def test_st_tabs_usage(self):
        """Verify st.tabs is used for tab navigation."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            content = f.read()
        
        assert 'viz_tabs = st.tabs([' in content, "Should use st.tabs for navigation"
        assert 'with viz_tabs[0]:' in content, "Should access viz_tabs by index"
    
    def test_plotly_imports(self):
        """Verify Plotly is properly imported in try blocks."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # Check for plotly imports
        assert 'import plotly.graph_objects as go' in content, "Should import plotly.graph_objects as go"
        assert 'import plotly.express as px' in content, "Should import plotly.express as px"
    
    def test_streamlit_plotly_usage(self):
        """Verify Plotly charts are rendered via st.plotly_chart."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            content = f.read()
        
        assert 'st.plotly_chart' in content, "Should use st.plotly_chart for Plotly charts"


class TestSyntaxValidation:
    """Validate Python syntax at test level."""
    
    def test_ui_syntax_valid(self):
        """Verify ui/app.py has valid Python syntax."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            content = f.read()
        
        # This will raise SyntaxError if invalid
        compile(content, ui_path, 'exec')
    
    def test_syntax_no_truncated_blocks(self):
        """Verify no try blocks are truncated (check for missing except blocks)."""
        ui_path = '/home/columbo/ExtData/AIprojects/RagDemo/ui/app.py'
        with open(ui_path, 'r') as f:
            lines = f.readlines()
        
        try_count = sum(1 for line in lines if 'try:' in line)
        except_count = sum(1 for line in lines if 'except' in line and ':' in line)
        
        assert try_count == except_count, f"Try-except imbalance: {try_count} try, {except_count} except"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
