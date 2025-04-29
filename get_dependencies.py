import json
import subprocess
import os
import shutil


def check_npm_installed():
    sss = shutil.which("npm") is not None
    return sss
    pass

if not check_npm_installed():
    print("❌ npm не найден. Установите Node.js и добавьте его в PATH.")
    exit(1)

# Получаем зависимости из package.json
def get_dependencies():
    try:
        with open('package.json', 'r', encoding='utf-8') as file:
            package_json = json.load(file)
        return list(package_json.get("dependencies", {}).keys())
    except FileNotFoundError:
        print("Файл package.json не найден.")
        return []
    except json.JSONDecodeError as e:
        print(f"Ошибка при чтении package.json: {e}")
        return []


# Получаем информацию о пакете
def fetch_package_info(package_name):
    try:
        result = subprocess.run(
            ['npm', 'view', package_name, '--json'],
            capture_output=True,
            text=True,
            check=True
        )
        package_info = json.loads(result.stdout)

        return {
            'name': package_info.get('name'),
            'version': package_info.get('version'),
            'description': package_info.get('description'),
            'license': package_info.get('license'),
            'time': package_info.get('time', {}).get('created', 'Неизвестно')
        }
    except subprocess.CalledProcessError as error:
        print(f'Ошибка при получении данных о пакете {package_name}: {error}')
        return None
    except json.JSONDecodeError as parse_error:
        print(f'Ошибка парсинга данных о пакете {package_name}: {parse_error}')
        return None


# Основная функция
def main():
    print('Сбор информации о зависимостях...')
    dependencies = get_dependencies()

    if not dependencies:
        print("Нет зависимостей для обработки.")
        return

    results = []
    for dep in dependencies:
        info = fetch_package_info(dep)
        if info:
            results.append(info)
        else:
            print(f"Не удалось получить информацию о пакете {dep}")

    if results:
        with open('dependencies-info.json', 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=2)
        print('Информация сохранена в dependencies-info.json')
    else:
        print("Не удалось собрать информацию о зависимостях.")


if __name__ == '__main__':
    main()
