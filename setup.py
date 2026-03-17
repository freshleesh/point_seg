from setuptools import find_packages, setup
from glob import glob

package_name = 'point_seg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc14',
    maintainer_email='jeongsangryu@gmail.com',
    description='Integrated high-speed point painting fusion node (YOLO seg + LiDAR projection).',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fusion_seg_paint_node = point_seg.fusion_seg_paint_node:main',
        ],
    },
)
