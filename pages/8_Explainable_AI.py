from dashboard_core import configure_page, inject_global_styles, render_explainable_ai_page, render_sidebar


configure_page("ECG AI Dashboard - Explainable AI")
inject_global_styles()
render_sidebar("Explainable AI")
render_explainable_ai_page()
