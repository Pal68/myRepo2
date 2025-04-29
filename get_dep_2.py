import json
import requests

output_file_name = './package_info2.txt'

def get_remaining_path(full_path):
    # Разделяем строку по 'node_modules/' и берем последнюю часть
    parts = full_path.split('node_modules/')
    return parts[-1] if len(parts) > 1 else full_path

def get_package_info(package_name):
    url = f"https://registry.npmjs.org/{package_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_dependencies_info(package_json_path):
    with open(package_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        packages = data.get('packages', {})
        # dependencies = packages.get('dependencies', {})
        # dev_dependencies = packages.get('devDependencies', {})

        # all_dependencies = {**dependencies, **dev_dependencies}
        all_dependencies = packages

        results = []
        with open(output_file_name, 'w', encoding='utf-8') as outfile:
            for i, package_name in enumerate(all_dependencies):
                package_name = 'node_modules/' + get_remaining_path(package_name)
                package_info = get_package_info(package_name[13:])
                if package_info:
                    print(f"{i}  {package_name[13:]} {package_info.get('version')}\n")
                    version = package_info.get('version', 'Версия не указана')
                    author = package_info.get('author', 'автор не указана')
                    resolved = package_info.get('tarball', 'Адрес не указан')
                    description = package_info.get('description')
                    time_create = package_info.get('time', {}).get('created', 'Неизвестно')
                    homepage = package_info.get('homepage', 'Домашняя страница не указана')
                    lic = package_info.get('license', 'Лицензия не указана')
                    results.append({
                        'npp': i,
                        'name': package_name,
                        'taraball': resolved,
                        'license': package_info.get('license', {}),
                        'description': package_info.get('description', 'No description available'),
                        'date': package_info.get('time', {}).get('created', 'Unknown')
                    })
                    outfile.write(f"  {package_name}; {version}; {resolved}; {time_create}; {lic}; {description}\n")
                else:
                    print(f"{i}  {package_name[13:]} не могу найти в инете")

            return results

if __name__ == "__main__":
    package_json_path = 'package-lock.json'
    dependencies_info = get_dependencies_info(package_json_path)
    for dependency in dependencies_info:
        print(f"Name: {dependency['name']}")
        print(f"License: {dependency['license']}")
        print(f"Description: {dependency['description']}")
        print(f"Date: {dependency['date']}")
        print(f"tarball: {dependency['tarball']}")
        print('-' * 50)
