from dashboard_core import configure_page, inject_global_styles, render_data_analysis_page, render_sidebar


configure_page("ECG AI Dashboard - Data Analysis")
inject_global_styles()
render_sidebar("Data Analysis")
render_data_analysis_page()
