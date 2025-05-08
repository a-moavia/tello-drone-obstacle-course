from setuptools import setup

package_name = 'drone_racer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sh0w0ff',
    maintainer_email='sh0w0ff@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = drone_racer.perception_node:main',
            'controller_node = drone_racer.controller_node:main',
            'controller_sm = drone_racer.controller_sm:main',

        ],
    },
)
