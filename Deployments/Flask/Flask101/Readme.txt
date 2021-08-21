#Flask Tutorial:

Flask - Microweb frame work for python.

Step 1. Setup Virtual Environment.
Anaconda prompt (run as administrator):
  a. Move to right directory
  b. pip install virtualenv --user
  b. virtualenv flasktutorial_env -p python3
  c. Activate virtual Environment
    LINUX - source flasktutorial_env/bin/activate
    WIN   - flasktutorial_env\Scripts\activate

Step 2. Install flask.
  a. pip install flask

-- Use pip freeze to check all packages installed
-- Use python -V to check version

Step 3. Create flask basic example.
  a. Create app.py

Step 4. Add your app.py in FLASK_APP Environment variable
   LINUX - export FLASK_APP=app.py
   WIN   - SET FLASK_APP=app.py

Step 5. run app!
python app.py


----------------------------------------------------------------------------
#Reloader
- Server watches all file changes and restarts server automatically

#Debugger
-A web based

#Help
flask --help
