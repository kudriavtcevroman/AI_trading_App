import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading_app', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='phone_number',
            field=models.CharField(blank=True, max_length=17, validators=[django.core.validators.RegexValidator(message="Номер телефона должен быть в формате: '+999999999'. До 15 цифр.", regex='^\\+?1?\\d{9,15}$')]),
        ),
    ]
