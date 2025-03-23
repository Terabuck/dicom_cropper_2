from django.urls import path
from .views import upload_dicom, serve_dicom_file, crop_dicom  # Add crop_dicom

urlpatterns = [
    path('upload/', upload_dicom, name='upload_dicom'),
    path('dicom/<str:filename>/', serve_dicom_file, name='serve_dicom_file'),
    path('media/outputs/<str:filename>/', serve_dicom_file, name='serve_output_dicom'),
    path('crop/', crop_dicom, name='crop_dicom'),  # Add this line
]