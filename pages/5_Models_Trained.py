from dashboard_core import configure_page, inject_global_styles, render_models_page, render_sidebar


configure_page("ECG AI Dashboard - Models Trained")
inject_global_styles()
render_sidebar("Models Trained")
render_models_page()
