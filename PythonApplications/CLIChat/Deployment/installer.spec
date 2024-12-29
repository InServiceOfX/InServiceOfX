a = Analysis(['clichat-installer/install.py'],
    datas=[
        ('clichat-installer/moregroq-0.1.0.tar.gz', '.'),
        ('clichat-installer/clichat-0.1.0.tar.gz', '.')
    ])

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='clichat-installer',
    debug=False,
    strip=False,
    upx=True,
    console=True)
