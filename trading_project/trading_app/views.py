from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm, PasswordChangeForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import UserProfile, UserAdditionalInfo
from .forms import CustomUserCreationForm, UserProfileForm, UserAdditionalInfoForm  # Форма для редактирования профиля

# Регистрация пользователя
def register_view(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # Сохраняем пользователя

            # Сохраняем обязательные данные пользователя
            user_profile = UserProfile.objects.create(
                user=user,
                first_name=form.cleaned_data.get('first_name'),
                last_name=form.cleaned_data.get('last_name'),
                email=form.cleaned_data.get('email'),
                phone_number=form.cleaned_data.get('phone_number'),
                birth_date=form.cleaned_data.get('birth_date'),
                gender=form.cleaned_data.get('gender')
            )

            # Сохраняем необязательные данные пользователя
            UserAdditionalInfo.objects.create(
                user_profile=user_profile,
                middle_name=form.cleaned_data.get('middle_name'),
                nickname=form.cleaned_data.get('nickname'),
                country=form.cleaned_data.get('country'),
                city=form.cleaned_data.get('city'),
                telegram=form.cleaned_data.get('telegram')
            )

            messages.success(request, f"Аккаунт {user.username} был успешно создан.")
            login(request, user)
            return redirect("profile")
    else:
        form = CustomUserCreationForm()
    return render(request, "register.html", {"form": form})

# Вход в систему
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("profile")  # Замените "home" на ваше целевое представление после входа
            else:
                messages.error(request, "Неправильное имя пользователя или пароль.")
        else:
            messages.error(request, "Неправильное имя пользователя или пароль.")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

# Выход из системы
def logout_view(request):
    logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("login")

# Личный кабинет - просмотр данных пользователя
@login_required(login_url='/login/')
def profile_view(request):
    user_profile = request.user.userprofile
    return render(request, 'profile.html', {'user_profile': user_profile})

# Редактирование данных пользователя
@login_required(login_url='/login/')
def profile_edit_view(request):
    user_profile = request.user.userprofile
    if request.method == "POST":
        form = UserProfileForm(request.POST, instance=user_profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Ваш профиль был обновлен.")
            return redirect('profile')
    else:
        form = UserProfileForm(instance=user_profile)
    return render(request, 'profile_edit.html', {'form': form})

# Смена пароля
@login_required(login_url='/login/')
def change_password_view(request):
    if request.method == "POST":
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # Обновляем сессию после смены пароля
            messages.success(request, "Пароль был успешно изменен.")
            return redirect('profile')
    else:
        form = PasswordChangeForm(user=request.user)
    return render(request, 'change_password.html', {'form': form})

# Удаление аккаунта
@login_required(login_url='/login/')
def delete_account_view(request):
    if request.method == "POST":
        request.user.delete()
        messages.success(request, "Ваш аккаунт был удален.")
        return redirect('login')
    return render(request, 'delete_account.html')

@login_required(login_url='/login/')
def edit_profile(request):
    user_profile = request.user.userprofile
    user_additional_info, created = UserAdditionalInfo.objects.get_or_create(user_profile=user_profile)

    if request.method == "POST":
        profile_form = UserProfileForm(request.POST, instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(request.POST, instance=user_additional_info)

        if profile_form.is_valid() and additional_info_form.is_valid():
            profile_form.save()
            additional_info_form.save()
            return redirect('profile')
    else:
        profile_form = UserProfileForm(instance=user_profile)
        additional_info_form = UserAdditionalInfoForm(instance=user_additional_info)

    return render(request, 'edit_profile.html', {
        'profile_form': profile_form,
        'additional_info_form': additional_info_form
    })