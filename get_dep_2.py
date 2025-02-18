import json
import requests

output_file_name = './package_info.txt'
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
                package_info = get_package_info(package_name[13:])
                print(f"{i}  {package_name[13:]} {package_info}")
                if package_info:
                    version = package_info.get('versions', 'Версия не указана')
                    author = package_info.get('author', 'автор не указана')
                    resolved = package_info.get('resolved', 'Адрес не указан')
                    description = package_info.get('description')
                    time_create = package_info.get('time', {}).get('created', 'Неизвестно')
                    homepage = package_info.get('homepage', 'Домашняя страница не указана')
                    lic = package_info.get('license', 'Лицензия не указана')
                    results.append({
                        'name': package_name,
                        'license': package_info.get('license', {}),
                        'description': package_info.get('description', 'No description available'),
                        'date': package_info.get('time', {}).get('created', 'Unknown')
                    })
                    outfile.write(f"  {package_name}; {version}; {resolved}; {time_create}; {lic}; {description}\n")

            return results

if __name__ == "__main__":
    package_json_path = 'package-lock.json'
    dependencies_info = get_dependencies_info(package_json_path)
    for dependency in dependencies_info:
        print(f"Name: {dependency['name']}")
        print(f"License: {dependency['license']}")
        print(f"Description: {dependency['description']}")
        print(f"Date: {dependency['date']}")
        print('-' * 50)