from dashboard_core import configure_page, inject_global_styles, render_home_page, render_sidebar


configure_page("ECG AI Dashboard - Home")
inject_global_styles()
render_sidebar("Home")
render_home_page()
