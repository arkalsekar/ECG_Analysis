from dashboard_core import configure_page, inject_global_styles, render_feature_extraction_page, render_sidebar


configure_page("ECG AI Dashboard - Feature Extraction")
inject_global_styles()
render_sidebar("Feature Extraction")
render_feature_extraction_page()
