from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from .models import UserProfile, UserAdditionalInfo
from datetime import date

class ProfileViewTests(TestCase):
    def setUp(self):
        # Создаем пользователя и его профиль
        self.user = User.objects.create_user(
            username='testuser',
            password='password123',
            email='testuser@example.com',
            first_name='Test',
            last_name='User'
        )
        self.user_profile = UserProfile.objects.create(
            user=self.user,
            phone_number='+1234567890',
            birth_date=date(2000, 1, 1),  # Используем объект date
            gender='M'
        )
        self.additional_info = UserAdditionalInfo.objects.create(
            user_profile=self.user_profile,
            nickname='Tester',
            country='Russia',
            city='Moscow',
            telegram='@tester'
        )
        self.client.login(username='testuser', password='password123')

    def test_profile_view_get(self):
        """
        Проверяем, что страница профиля отображается корректно
        """
        response = self.client.get(reverse('profile'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'profile/profile.html')
        self.assertContains(response, 'Test')  # Проверяем, что имя отображается

    def test_profile_update_success(self):
        """
        Тест успешного редактирования профиля
        """
        response = self.client.post(reverse('profile'), {
            'username': 'testuser_updated',
            'first_name': 'NewFirstName',
            'last_name': 'NewLastName',
            'email': 'newemail@example.com',
            'phone_number': '+9876543210',
            'birth_date': '1990-12-12',
            'gender': 'M',
            'nickname': 'NewNick',
            'country': 'USA',
            'city': 'New York',
            'telegram': '@newtelegram',
        })

        # Проверяем редирект на страницу профиля
        self.assertRedirects(response, reverse('profile'))

        # Обновляем данные пользователя
        self.user.refresh_from_db()
        self.user_profile.refresh_from_db()
        self.additional_info.refresh_from_db()

        # Проверяем, что данные были обновлены
        self.assertEqual(self.user.username, 'testuser_updated')
        self.assertEqual(self.user.first_name, 'NewFirstName')
        self.assertEqual(self.user.email, 'newemail@example.com')
        self.assertEqual(self.user_profile.phone_number, '+9876543210')
        self.assertEqual(self.user_profile.birth_date.strftime('%Y-%m-%d'), '1990-12-12')
        self.assertEqual(self.additional_info.nickname, 'NewNick')

    def test_profile_update_validation_error(self):
        """
        Тест неуспешного редактирования профиля (ошибка валидации)
        """
        response = self.client.post(reverse('profile'), {
            'username': '',  # Некорректный логин
            'first_name': 'NewFirstName',
            'last_name': 'NewLastName',
            'email': 'invalid_email',  # Некорректный email
            'phone_number': 'invalid_phone',  # Некорректный номер
            'birth_date': '2030-12-12',  # Некорректная дата рождения (в будущем)
        })

        # Проверяем, что не произошло перенаправления
        self.assertEqual(response.status_code, 200)

        # Проверяем наличие сообщений об ошибке
        self.assertFormError(response, 'user_form', 'username', 'Обязательное поле.')
        self.assertFormError(response, 'user_form', 'email', 'Введите правильный адрес электронной почты.')
        self.assertFormError(response, 'profile_form', 'phone_number', 'Введите корректный номер телефона в формате: \'+999999999\'.')
        self.assertFormError(response, 'profile_form', 'birth_date', 'Дата рождения не может быть в будущем.')

    def test_profile_update_with_long_values(self):
        """
        Тест на слишком длинные значения в полях формы (проверка длины).
        """
        long_string = 'a' * 151  # Превышает максимальную длину для логина (150 символов)

        response = self.client.post(reverse('profile'), {
            'username': long_string,
            'first_name': 'a' * 31,  # Максимальная длина 30 символов
            'last_name': 'a' * 31,
            'email': 'newemail@example.com',
            'phone_number': '+9876543210',
            'birth_date': '1990-12-12',
            'gender': 'M',
            'nickname': 'NewNick',
            'country': 'USA',
            'city': 'New York',
            'telegram': '@newtelegram',
        })

        self.assertEqual(response.status_code, 200)  # Остаемся на странице из-за ошибки

        # Проверяем ошибки формы
        form = response.context['user_form']
        self.assertTrue(
            form.errors['username'][0].startswith('Убедитесь, что это значение содержит не более 150 символов'),
            "Expected a different error message for 'username' field."
        )
        self.assertTrue(
            form.errors['first_name'][0].startswith('Убедитесь, что это значение содержит не более 30 символов'),
            "Expected a different error message for 'first_name' field."
        )
        self.assertTrue(
            form.errors['last_name'][0].startswith('Убедитесь, что это значение содержит не более 30 символов'),
            "Expected a different error message for 'last_name' field."
        )

    def test_profile_update_with_sql_injection(self):
        """
        Тест на SQL-инъекцию через поля формы.
        """
        sql_injection_string = "'; DROP TABLE users; --"

        response = self.client.post(reverse('profile'), {
            'username': sql_injection_string,  # Попытка SQL-инъекции в логин
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'testuser@example.com',
            'phone_number': '+1234567890',
            'birth_date': '2000-01-01',
            'gender': 'M',
            'nickname': 'Tester',
            'country': 'Russia',
            'city': 'Moscow',
            'telegram': '@tester',
        })

        # Проверяем, что SQL-инъекция не выполнена и приложение не упало
        self.assertEqual(response.status_code, 200)

        # Проверяем ошибки формы
        form = response.context['user_form']
        self.assertTrue(
            form.errors['username'][0].startswith('Введите правильное имя пользователя'),
            "Expected a different error message for 'username' field."
        )

        # Дополнительно можно проверить, что другой пользователь не был изменен
        another_user = User.objects.filter(username='anotheruser').first()
        if another_user:
            self.assertEqual(another_user.username, 'anotheruser')
            self.assertEqual(another_user.email, 'anotheruser@example.com')

    def test_profile_update_with_no_changes(self):
        """
        Тест на отсутствие изменений в профиле (отправка формы без изменений).
        """
        response = self.client.post(reverse('profile'), {
            'username': self.user.username,  # Все данные остаются такими же
            'first_name': self.user.first_name,
            'last_name': self.user.last_name,
            'email': self.user.email,
            'phone_number': self.user_profile.phone_number,
            'birth_date': self.user_profile.birth_date.strftime('%Y-%m-%d'),
            'gender': self.user_profile.gender,
            'nickname': self.additional_info.nickname,
            'country': self.additional_info.country,
            'city': self.additional_info.city,
            'telegram': self.additional_info.telegram,
        })

        # Проверяем, что произошло перенаправление (успешное обновление)
        self.assertRedirects(response, reverse('profile'))

        # Проверяем, что данные не изменились
        self.user.refresh_from_db()
        self.assertEqual(self.user.username, 'testuser')  # Проверяем, что логин остался прежним
        self.assertEqual(self.user.first_name, 'Test')  # Проверяем, что имя не изменилось

    def test_profile_update_another_user(self):
        """
        Тест, который проверяет, что нельзя редактировать профиль другого пользователя.
        """
        another_user = User.objects.create_user(
            username='anotheruser',
            password='password123',
            email='anotheruser@example.com'
        )

        response = self.client.post(reverse('profile'), {
            'username': 'testuser',
            'first_name': 'NewFirstName',
            'last_name': 'NewLastName',
            'email': 'newemail@example.com',
            'phone_number': '+9876543210',
            'birth_date': '1990-12-12',
            'gender': 'M',
            'nickname': 'NewNick',
            'country': 'USA',
            'city': 'New York',
            'telegram': '@newtelegram',
            'user': another_user.id  # Попытка подмены ID пользователя
        })

        # Проверяем, что данные другого пользователя не были изменены
        another_user.refresh_from_db()
        self.assertEqual(another_user.username, 'anotheruser')
        self.assertEqual(another_user.email, 'anotheruser@example.com')
