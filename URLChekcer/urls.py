from django.urls import path
from . import views

urlpatterns = [
    path("run-code/", views.run_code, name="run_code"),
    path("execute_code/", views.execute_code, name="execute_code")
]
