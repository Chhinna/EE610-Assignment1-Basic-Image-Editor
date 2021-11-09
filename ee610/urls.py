from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name='home'),
    path('make_negative',views.make_negative, name='make_negative'),
    path('log_transform',views.log_transform, name='log_transform'),
    path('gamma_correct',views.gamma_correct, name='gamma_correct'),
    path('blur',views.blur, name='blur'),
    path('sharp',views.sharp, name='sharp'),
    path('hist_equal',views.hist_equal, name='hist_equal'),
]