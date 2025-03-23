from django.urls import path
from . import views
from .views import upload_dicom, serve_dicom_file, crop_dicom, clear_outputs, dicom_upload_view  # Add crop_dicom

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', upload_dicom, name='upload_dicom'),
    path('dicom-upload/', dicom_upload_view, name='dicom-upload'),
    path('dicom/<str:filename>/', serve_dicom_file, name='serve_dicom_file'),
    path('media/outputs/<str:filename>/', serve_dicom_file, name='serve_output_dicom'),
    path('crop/', crop_dicom, name='crop_dicom'),
    path('clear-outputs/', clear_outputs, name='clear_outputs'),
]
