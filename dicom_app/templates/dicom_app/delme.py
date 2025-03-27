@csrf_exempt
def crop_dicom(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dicom_path = os.path.join(settings.MEDIA_ROOT, 'uploads/temp.dcm')
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array
            original_rows = ds.Rows
            original_columns = ds.Columns

            if 'type' in data and data['type'] == 'poly':
                # Polygon handling (existing code)
                if 'points' not in data or len(data['points']) < 3:
                    return JsonResponse({'error': 'Polygon needs at least 3 points'}, status=400)
                
                for p in data['points']:
                    if not (0 <= p['x'] < ds.Columns and 0 <= p['y'] < ds.Rows):
                        return JsonResponse({'error': 'Point out of bounds'}, status=400)
                
                mask = create_polygon_mask(ds.Columns, ds.Rows, data['points'])
                masked_array = pixel_array * mask
                ds.PixelData = masked_array.tobytes()

            else:
                # Rectangle handling with centering
                x1 = max(0, data['x1'])
                y1 = max(0, data['y1'])
                x2 = min(data['x2'], ds.Columns)
                y2 = min(data['y2'], ds.Rows)
                
                cropped_region = pixel_array[y1:y2, x1:x2]
                cropped_height, cropped_width = cropped_region.shape

                # Create centered array with original dimensions
                centered_array = np.zeros((original_rows, original_columns), dtype=pixel_array.dtype)
                start_y = (original_rows - cropped_height) // 2
                start_x = (original_columns - cropped_width) // 2
                end_y = start_y + cropped_height
                end_x = start_x + cropped_width

                # Ensure bounds
                start_y = max(0, start_y)
                start_x = max(0, start_x)
                end_y = min(original_rows, end_y)
                end_x = min(original_columns, end_x)

                centered_array[start_y:end_y, start_x:end_x] = cropped_region
                ds.PixelData = centered_array.tobytes()

            # Maintain original dimensions
            ds.Rows = original_rows
            ds.Columns = original_columns

            # Save and return
            output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            cropped_filename = f'cropped_{uuid.uuid4().hex}.dcm'
            cropped_path = os.path.join(output_dir, cropped_filename)
            ds.save_as(cropped_path)
            
            return JsonResponse({'cropped_url': f'/media/outputs/{cropped_filename}'})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)