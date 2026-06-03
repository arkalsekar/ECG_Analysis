from dashboard_core import configure_page, inject_global_styles, render_live_prediction_page, render_sidebar


configure_page("ECG AI Dashboard - Live Prediction")
inject_global_styles()
render_sidebar("Live Prediction")
render_live_prediction_page()
