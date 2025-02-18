import json


output_file_name = './package_info.txt'
try:
    with open('package-lock.json', 'r', encoding='utf-8') as f:
        package_lock = json.load(f)

    packages = package_lock.get('packages', {})


    print("Информация о пакетах:")
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        for package_name, package_info in packages.items():
            if package_name == "":
                continue # Пропускаем корневой пакет
            version = package_info.get('version', 'Версия не указана')
            resolved = package_info.get('resolved', 'Адрес не указан')
            homepage = package_info.get('homepage', 'Домашняя страница не указана')
            lic = package_info.get('license', 'Лицензия не указана')

            print(f"  Имя: {package_name}")
            print(f"  Версия: {version}")
            print(f"  Адрес: {resolved}")
            print(f"  Домашняя страница: {homepage}")
            print(f"  Лицензия: {lic}")
            print("---")


            outfile.write(f"  {package_name}; {version}; {resolved}; {homepage}; {lic} \n")


except FileNotFoundError:
    print("Ошибка: файл package-lock.json не найден.")
except json.JSONDecodeError:
    print("Ошибка: Некорректный формат JSON в файле package-lock.json.")
except Exception as e:
    print(f"Произошла ошибка: {e}")