from django import forms

# Existing form for single DICOM file upload
class DicomUploadForm(forms.Form):
    dicom_file = forms.FileField(label="Upload DICOM File")

# New form for selecting a folder
class DicomFolderForm(forms.Form):
    folder_path = forms.CharField(
        label="Select Folder",
        widget=forms.TextInput(attrs={'readonly': 'readonly'}),  # Make it read-only
        required=True
    )