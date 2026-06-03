from dashboard_core import configure_page, inject_global_styles, render_best_model_page, render_sidebar


configure_page("ECG AI Dashboard - Best Model")
inject_global_styles()
render_sidebar("Best Model")
render_best_model_page()
