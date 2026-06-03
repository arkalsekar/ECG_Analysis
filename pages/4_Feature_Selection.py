from dashboard_core import configure_page, inject_global_styles, render_feature_selection_page, render_sidebar


configure_page("ECG AI Dashboard - Feature Selection")
inject_global_styles()
render_sidebar("Feature Selection")
render_feature_selection_page()
