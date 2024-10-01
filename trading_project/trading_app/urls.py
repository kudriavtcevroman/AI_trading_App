from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home_view, name='home'),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path('profile/', views.profile_view, name='profile'),
    path('profile/delete/', views.delete_account_view, name='delete_account'),
    path('password_change/', auth_views.PasswordChangeView.as_view(template_name='password_change.html'), name='password_change'),
    path('password_change_done/', auth_views.PasswordChangeDoneView.as_view(template_name='password_change_done.html'), name='password_change_done'),
    path('get_available_dates/', views.get_available_dates, name='get_available_dates'),
    path('upload_training_data/', views.upload_training_data, name='upload_training_data'),
    path('delete_table/<int:table_id>/', views.delete_table, name='delete_table'),
    path('rename_table/<int:table_id>/', views.upload_training_data, name='rename_table'),  # Для изменения названия
    path('view_table/<int:table_id>/', views.view_table, name='view_table'),  # Для просмотра таблицы
]
