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
    path('training/', views.training_home, name='training_home'),
    path('training/upload_data/', views.upload_training_data, name='upload_training_data'),
    path('training/training_model/', views.training_model, name='training_model'),
    path('password_change/', auth_views.PasswordChangeView.as_view(template_name='password_change.html'), name='password_change'),
    path('password_change_done/', auth_views.PasswordChangeDoneView.as_view(template_name='password_change_done.html'), name='password_change_done'),
    path('delete_dataset/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
    path('rename_dataset/<int:dataset_id>/', views.rename_dataset, name='rename_dataset'),
    path('view_dataset/<int:dataset_id>/', views.view_dataset, name='view_dataset'),
    path('get_date_range/', views.get_date_range, name='get_date_range'),
]
