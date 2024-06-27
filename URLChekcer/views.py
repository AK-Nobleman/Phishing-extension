from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from URLChekcer.utils import run_python_code
import json
import logging
# views.py
@csrf_exempt
# Create your views here.
def run_code(request):
    if request.method == 'POST':
        try:
            # Retrieve input_url from the request body
            data = json.loads(request.body)
            input_url = data.get('url', '')

            # Process the input_url if needed
            value, accuracy = run_python_code(input_url)
    
            # Prepare the response data
            result = {"status": "success", "data": f"Prediction: {value}            Accuracy: {accuracy:.4f}"}
            
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    if request.method == 'GET':
        return JsonResponse({"status": "error", "message": "GETs"}, status=400)
    return JsonResponse({"status": "error", "message": request.method}, status=400)

        

def execute_code(request):
    output = run_python_code()
    
    context = {'output': output}

    # Render template with output
    return render(request, 'output_template.html', context)

    


def show_link(request):
    if request.method == 'POST':
        data = request.json()
        input_link = data.get('url', '')
        
        # Process the input_link as needed

        return JsonResponse({'input_link': input_link})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
