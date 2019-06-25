from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

#from flaskr.auth import login_required
#from flaskr.db import get_db

bp = Blueprint('project', __name__)


@bp.route('/')
def index():
    #db = get_db()
    # posts = db.execute(
    #    'SELECT p.id, title, body, created, author_id, username'
    #    ' FROM post p JOIN user u ON p.author_id = u.id'
    #    ' ORDER BY created DESC'
    # ).fetchall()
    # substitute posts with whatever in the db
    return render_template('project/index.html', posts=None)
