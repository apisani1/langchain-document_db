SQLite format 3   @    ¬   Ù                                                           ¬ .v   î î                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ##etablefile_errorsfile_errorsCREATE TABLE file_errors (
	id INTEGER NOT NULL, 
	timestamp DATETIME, 
	root VARCHAR(255), 
	file VARCHAR(255), 
	file_extension VARCHAR(50), 
	error_type VARCHAR(100), 
	error_message TEXT, 
	error_traceback TEXT, 
	PRIMARY KEY (id)
)   ÕA    ÙûöñìçâÝØÓÎÉÄ¿ºµ°«¦¡~ytoje`[VQLGB=83.)$ü÷òíèãÞÙÔÏÊÅ¿¹³­§¡}wqke_YSMGA;5/)#ÿùóíçáÛÕÏÉÃ½·±«¥{uoic]WQKE?93-'!	ý÷ñëåßÙÓÍÇÁ»µ¯©£ysmga[UOIC=71+%ûõïéãÝ×ÑËÅ¿¹³­§¡}wqke_YSMGA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Ø*   ×(   Ö&   Õ$   Ô"   Ó    Ò   Ñ   Ð   Ï   Î   Í   Ì   Ë   Ê   É   È
   Ç   Æ   Å   Ä   Ã    Â~   Á|   Àz   ¿x   ¾v   ½t   ¼s   »r   ºp   ¹n   ¸l   ·j   ¶h   µf   ´d   ³b   ²`   °^   ¯[   ®Y   ­W   ¬U   «S   ªQ   ©O   ¨M   §K   ¦I   ¥G   ¤E   £C   ¢A   ¡?    =   ;   9   7   5   3   1   /   -   +   )   '   %   #   !                                    	                  }   {   ~y   }w   |u   {s   zq   yo   xm   wk   vi   ug   te   sc   ra   q_   p]   o[   nY   mW   lU   kS   jQ   iO   hM   gK   fI   eG   dE   cC   bA   a?   `=   _;   ^9   ]7   \5   [3   Z1   Y/   X-   W+   V)   U'   T%   S#   R!   Q   P   O   N   M   L   K   J   I   H   G   F	   E   D   C   B   A   @}   ?{   >y   =w   <u   ;s   :q   9o   8m   7k   6i   5g   4e   3c   2a   1_   0]   /[   .Y   -W   ,U   +S   *Q   )O   (M   'K   &I   %G   $E   #C   "A   !?    =   ;   9   7   5   3   1   /   -   +   )   '   %   #   !                        
   	      
                ï                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               a A=!Yk2024-08-02 20:00:38.854083/Users/antonio/DocumentsIconValueErrorInvalid file /Users/antonio/Documents/Icon. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Icon. The FileType.UNK file type is not supported in partition.
 A=#}2024-08-02 20:00:20.164090/Users/antonio/Documentshotmail.csvcsvTypeErrorCSVLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: CSVLoader.__init__() got an unexpected keyword argument 'on_error'
   Õ 	÷Õ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          A=A2024-08-02 20:00:39.074445/Users/antonio/Documentslogin_gov_backup_codes.txttxtTypeErrorTextLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: TextLoader.__init__() got an unexpected keyword argument 'on_error'
 A=+!m2024-08-02 20:00:38.857226/Users/antonio/DocumentsMy Lens.numbersnumbersValueErrorInvalid file /Users/antonio/Documents/My Lens.numbers. The FileType.ZIP file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/My Lens.numbers. The FileType.ZIP file type is not supported in partition.
   Ø 	æØ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             A=1!s2024-08-02 20:00:49.312246/Users/antonio/DocumentsAntonio Pisani.vcfvcfValueErrorInvalid file /Users/antonio/Documents/Antonio Pisani.vcf. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Antonio Pisani.vcf. The FileType.UNK file type is not supported in partition.
 A=9!{2024-08-02 20:00:41.234592/Users/antonio/DocumentsAntonio's Notebook.urlurlValueErrorInvalid file /Users/antonio/Documents/Antonio's Notebook.url. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Antonio's Notebook.url. The FileType.UNK file type is not supported in partition.
   Å ñÅ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ) A=E!2024-08-02 20:01:42.357754/Users/antonio/Documentselectronic business card.vcfvcfValueErrorInvalid file /Users/antonio/Documents/electronic business card.vcf. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/electronic business card.vcf. The FileType.UNK file type is not supported in partition.
 A=}2024-08-02 20:01:40.775665/Users/antonio/Documentsgmail.csvcsvTypeErrorCSVLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: CSVLoader.__init__() got an unexpected keyword argument 'on_error'
   ê 	æê                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              y
 A=%!gy2024-08-02 20:01:44.359889/Users/antonio/Documentswindows7.pfxpfxValueErrorInvalid file /Users/antonio/Documents/windows7.pfx. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/windows7.pfx. The FileType.UNK file type is not supported in partition.
	 A=7!y2024-08-02 20:01:42.911301/Users/antonio/DocumentsBlurb Magazine.saprojsaprojValueErrorInvalid file /Users/antonio/Documents/Blurb Magazine.saproj. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Blurb Magazine.saproj. The FileType.UNK file type is not supported in partition.
   ¤ ß¤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        8 A=O!#2024-08-02 20:01:45.425434/Users/antonio/DocumentsGhassan Abboud and 734 others.vcfvcfValueErrorInvalid file /Users/antonio/Documents/Ghassan Abboud and 734 others.vcf. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Ghassan Abboud and 734 others.vcf. The FileType.UNK file type is not supported in partition.
 A=?2024-08-02 20:01:44.395897/Users/antonio/Documentsgithub-recovery-codes.txttxtTypeErrorTextLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: TextLoader.__init__() got an unexpected keyword argument 'on_error'
   M 	®M                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^ Aq1!'92024-08-02 20:02:27.449024/Users/antonio/Documents/Unreal Projects/MyProjectMyProject.uprojectuprojectValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/MyProject.uproject. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/MyProject.uproject. The FileType.UNK file type is not supported in partition.
O Ac7!12024-08-02 20:01:49.519661/Users/antonio/Documents/Affinity PublisherResume Template.afpubafpubValueErrorInvalid file /Users/antonio/Documents/Affinity Publisher/Resume Template.afpub. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Affinity Publisher/Resume Template.afpub. The FileType.UNK file type is not supported in partition.
    µj                                                                                                                                                                                                                                                    e A+!/A2024-08-02 20:02:28.474477/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultGame.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultGame.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultGame.ini. The FileType.UNK file type is not supported in partition.
H A;2024-08-02 20:02:28.470156/Users/antonio/Documents/Unreal Projects/MyProject/Saved/AutosavesPackageRestoreData.jsonjsonTypeErrorJSONLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: JSONLoader.__init__() got an unexpected keyword argument 'on_error'
H A!+2024-08-02 20:02:28.467805/Users/antonio/Documents/Unreal Projects/MyProject/Saved/Config/WorldState1586765442.jsonjsonTypeErrorJSONLoader.__init__() got an unexpected keyword argument 'on_error'Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 78, in load_document_lazy
    loader = _get_document_loader(file_path, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 46, in _get_document_loader
    return loader(file_path, **loader_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: JSONLoader.__init__() got an unexpected keyword argument 'on_error'
   Ð 	zÐ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ' AW![m2024-08-02 20:02:28.479014/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultVirtualProductionUtilities.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultVirtualProductionUtilities.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultVirtualProductionUtilities.ini. The FileType.UNK file type is not supported in partition.
 A?!CU2024-08-02 20:02:28.476368/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultEditorSettings.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEditorSettings.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEditorSettings.ini. The FileType.UNK file type is not supported in partition.
   ' 	'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           h A-!1C2024-08-02 20:02:28.484336/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultInput.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultInput.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultInput.ini. The FileType.UNK file type is not supported in partition.
k A/!3E2024-08-02 20:02:28.482672/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultEngine.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEngine.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEngine.ini. The FileType.UNK file type is not supported in partition.
    	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               t A5!9K2024-08-02 20:02:28.489145/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultLightmass.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultLightmass.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultLightmass.ini. The FileType.UNK file type is not supported in partition.
k A/!3E2024-08-02 20:02:28.487013/Users/antonio/Documents/Unreal Projects/MyProject/ConfigDefaultEditor.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEditor.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/DefaultEditor.ini. The FileType.UNK file type is not supported in partition.
   ø 	yø                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ~ A7!=O2024-08-02 20:02:28.499384/Users/antonio/Documents/Unreal Projects/MyProject/ContentMain_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Main_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Main_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
 A/!CU2024-08-02 20:02:28.494561/Users/antonio/Documents/Unreal Projects/MyProject/Config/WindowsWindowsEngine.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/Windows/WindowsEngine.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Config/Windows/WindowsEngine.ini. The FileType.UNK file type is not supported in partition.
   î 	¥î                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  4 A%7!as2024-08-02 20:02:28.530623/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsMain_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Main_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Main_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
X A!%72024-08-02 20:02:28.504626/Users/antonio/Documents/Unreal Projects/MyProject/ContentMain.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Main.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Main.umap. The FileType.UNK file type is not supported in partition.
   £ 	o£                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       I A%E!o2024-08-02 20:02:28.549337/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsVR-Scouting_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/VR-Scouting_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/VR-Scouting_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
 A%!I[2024-08-02 20:02:28.536608/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsVcam.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Vcam.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Vcam.umap. The FileType.UNK file type is not supported in partition.
     	=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      A%'!Qc2024-08-02 20:02:28.564721/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsLiveComp.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/LiveComp.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/LiveComp.umap. The FileType.UNK file type is not supported in partition.
@ A%?!i{2024-08-02 20:02:28.562787/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsLiveComp_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/LiveComp_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/LiveComp_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
   ¬ 	c¬                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                4! A%7!as2024-08-02 20:02:28.567950/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsVcam_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Vcam_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Vcam_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
  A%'!Qc2024-08-02 20:02:28.566403/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsNdisplay.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Ndisplay.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Ndisplay.umap. The FileType.UNK file type is not supported in partition.
    	=                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ## A%-!Wi2024-08-02 20:02:28.575832/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsVR-Scouting.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/VR-Scouting.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/VR-Scouting.umap. The FileType.UNK file type is not supported in partition.
@" A%?!i{2024-08-02 20:02:28.569537/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsNdisplay_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Ndisplay_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Ndisplay_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
   ¸ 	o¸                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            4% A/-!as2024-08-02 20:02:28.584931/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MaterialsMI_Chrome.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Chrome.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Chrome.uasset. The FileType.UNK file type is not supported in partition.
$ A%!I[2024-08-02 20:02:28.577665/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MapsMain.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Main.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Maps/Main.umap. The FileType.UNK file type is not supported in partition.
    	C                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                4' A/-!as2024-08-02 20:02:28.597146/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MaterialsM_Backing.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/M_Backing.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/M_Backing.uasset. The FileType.UNK file type is not supported in partition.
:& A/1!ew2024-08-02 20:02:28.591889/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MaterialsM_Live_Comp.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/M_Live_Comp.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/M_Live_Comp.uasset. The FileType.UNK file type is not supported in partition.
    	L                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       F) A/9!m2024-08-02 20:02:28.605114/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MaterialsMI_Backing_Inst.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Backing_Inst.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Backing_Inst.uasset. The FileType.UNK file type is not supported in partition.
1( A/+!_q2024-08-02 20:02:28.603473/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MaterialsMI_Floor.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Floor.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Materials/MI_Floor.uasset. The FileType.UNK file type is not supported in partition.
   q 	7q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     C+ A+;!k}2024-08-02 20:02:28.609290/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOBM_Media_Profile.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/BM_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/BM_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.
F* A+=!m2024-08-02 20:02:28.607842/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOAJA_Media_Profile.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/AJA_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/AJA_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.
   t 	4t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        =- A+7!gy2024-08-02 20:02:28.612134/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOMediaSource-02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaSource-02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaSource-02.uasset. The FileType.UNK file type is not supported in partition.
I, A+?!o2024-08-02 20:02:28.610682/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOFile_Media_Profile.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/File_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/File_Media_Profile.uasset. The FileType.UNK file type is not supported in partition.
    	@                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    =/ A+7!gy2024-08-02 20:02:28.618922/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOMediaBundle-02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.
=. A+7!gy2024-08-02 20:02:28.616803/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOMediaBundle-01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.
    	@                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    =1 A+7!gy2024-08-02 20:02:28.624541/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOMediaSource-01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaSource-01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaSource-01.uasset. The FileType.UNK file type is not supported in partition.
=0 A+7!gy2024-08-02 20:02:28.622520/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIOMediaOutput-01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaOutput-01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaOutput-01.uasset. The FileType.UNK file type is not supported in partition.
    ³                                                                                                                                                                                                                                                                                                                                                                                                 #3 AaE!+=2024-08-02 20:02:28.627545/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssetsMediaP_MediaBundle-01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/MediaP_MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/MediaP_MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.
J2 Aa_!EW2024-08-02 20:02:28.626148/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssetsRT_MediaBundle-01_LensDisplacement.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/RT_MediaBundle-01_LensDisplacement.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/RT_MediaBundle-01_LensDisplacement.uasset. The FileType.UNK file type is not supported in partition.
   Æ æÆ                                                                                                                                                                                                                                                                                                                                                                                                                                                          5 AaA!'92024-08-02 20:02:28.642165/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssetsT_MediaBundle-01_BC.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/T_MediaBundle-01_BC.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/T_MediaBundle-01_BC.uasset. The FileType.UNK file type is not supported in partition.
4 Aa=!#52024-08-02 20:02:28.638409/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssetsMI_MediaBundle-01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/MI_MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-01_InnerAssets/MI_MediaBundle-01.uasset. The FileType.UNK file type is not supported in partition.
    à                                                                                                                                                                                                                                                                                                                                                                                                       J7 Aa_!EW2024-08-02 20:02:28.654957/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssetsRT_MediaBundle-02_LensDisplacement.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/RT_MediaBundle-02_LensDisplacement.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/RT_MediaBundle-02_LensDisplacement.uasset. The FileType.UNK file type is not supported in partition.
6 AaA!'92024-08-02 20:02:28.643927/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssetsT_MediaBundle-02_BC.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/T_MediaBundle-02_BC.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/T_MediaBundle-02_BC.uasset. The FileType.UNK file type is not supported in partition.
   À æÀ                                                                                                                                                                                                                                                                                                                                                                                                                                                    #9 AaE!+=2024-08-02 20:02:28.660995/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssetsMediaP_MediaBundle-02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/MediaP_MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/MediaP_MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.
8 Aa=!#52024-08-02 20:02:28.658319/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssetsMI_MediaBundle-02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/MI_MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/MediaIO/MediaBundle-02_InnerAssets/MI_MediaBundle-02.uasset. The FileType.UNK file type is not supported in partition.
    	U                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         =; A-5!gy2024-08-02 20:02:28.711088/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/GeometrySM_Floor_Disk.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Geometry/SM_Floor_Disk.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Geometry/SM_Floor_Disk.uasset. The FileType.UNK file type is not supported in partition.
(: A-'!Yk2024-08-02 20:02:28.685787/Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/GeometrySM_Cyc.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Geometry/SM_Cyc.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/VprodProject/Geometry/SM_Cyc.uasset. The FileType.UNK file type is not supported in partition.
   ¾ 	|¾                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ;= A)9!gy2024-08-02 20:02:28.728480/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsAdvanced_Lighting.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Advanced_Lighting.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Advanced_Lighting.umap. The FileType.UNK file type is not supported in partition.
< A-!AS2024-08-02 20:02:28.721733/Users/antonio/Documents/Unreal Projects/MyProject/Content/MoviesMediaExample.mp4mp4ValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Movies/MediaExample.mp4. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/Movies/MediaExample.mp4. The FileType.UNK file type is not supported in partition.
    	W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            L? A)C!q2024-08-02 20:02:28.794258/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsStarterMap_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/StarterMap_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/StarterMap_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
&> A)+!Yk2024-08-02 20:02:28.776311/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsStarterMap.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/StarterMap.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/StarterMap.umap. The FileType.UNK file type is not supported in partition.
   > 	>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [A A)M!{2024-08-02 20:02:28.814201/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsMinimal_Default_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Minimal_Default_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Minimal_Default_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
a@ A)Q!2024-08-02 20:02:28.809408/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsAdvanced_Lighting_BuiltData.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Advanced_Lighting_BuiltData.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Advanced_Lighting_BuiltData.uasset. The FileType.UNK file type is not supported in partition.
   a 	Ha                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     dC A)S!2024-08-02 20:02:28.823131/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/HDRIHDRI_Epic_Courtyard_Daylight.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/HDRI/HDRI_Epic_Courtyard_Daylight.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/HDRI/HDRI_Epic_Courtyard_Daylight.uasset. The FileType.UNK file type is not supported in partition.
5B A)5!cu2024-08-02 20:02:28.816990/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MapsMinimal_Default.umapumapValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Minimal_Default.umap. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Maps/Minimal_Default.umap. The FileType.UNK file type is not supported in partition.
   } 	4}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 4E A1+!as2024-08-02 20:02:28.827516/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Bush_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Bush_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Bush_D.uasset. The FileType.UNK file type is not supported in partition.
ID A19!o2024-08-02 20:02:28.825698/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ground_Moss_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Moss_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Moss_N.uasset. The FileType.UNK file type is not supported in partition.
   t 	1t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        :G A1/!ew2024-08-02 20:02:28.831574/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Statue_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Statue_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Statue_M.uasset. The FileType.UNK file type is not supported in partition.
LF A1;!q2024-08-02 20:02:28.829682/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Detail_Rocky_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Detail_Rocky_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Detail_Rocky_N.uasset. The FileType.UNK file type is not supported in partition.
   b 	%b                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      @I A13!i{2024-08-02 20:02:28.836408/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_RockMesh_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_RockMesh_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_RockMesh_M.uasset. The FileType.UNK file type is not supported in partition.
XH A1C!y2024-08-02 20:02:28.833439/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Hewn_Stone_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_M.uasset. The FileType.UNK file type is not supported in partition.
   V 	+V                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          RK A1?!u2024-08-02 20:02:28.843551/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_New_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_N.uasset. The FileType.UNK file type is not supported in partition.
RJ A1?!u2024-08-02 20:02:28.840860/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_Old_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Old_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Old_D.uasset. The FileType.UNK file type is not supported in partition.
   Y 	1Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             UM A1A!w	2024-08-02 20:02:28.846585/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Poured_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Poured_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Poured_D.uasset. The FileType.UNK file type is not supported in partition.
LL A1;!q2024-08-02 20:02:28.845034/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Perlin_Noise_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Perlin_Noise_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Perlin_Noise_M.uasset. The FileType.UNK file type is not supported in partition.
    	:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       4O A1+!as2024-08-02 20:02:28.850683/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Lamp_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Lamp_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Lamp_M.uasset. The FileType.UNK file type is not supported in partition.
CN A15!k}2024-08-02 20:02:28.849080/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Pine_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Pine_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Pine_D.uasset. The FileType.UNK file type is not supported in partition.
   e 	:e                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         RQ A1?!u2024-08-02 20:02:28.853876/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Tiles_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_D.uasset. The FileType.UNK file type is not supported in partition.
CP A15!k}2024-08-02 20:02:28.852180/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Smoke_SubUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Smoke_SubUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Smoke_SubUV.uasset. The FileType.UNK file type is not supported in partition.
   S 	4S                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       ^S A1G!}2024-08-02 20:02:28.857256/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Smooth_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_N.uasset. The FileType.UNK file type is not supported in partition.
IR A19!o2024-08-02 20:02:28.855539/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Basalt_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Basalt_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Basalt_N.uasset. The FileType.UNK file type is not supported in partition.
   A 	A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [U A1E!{2024-08-02 20:02:28.884441/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Floor_Walnut_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_M.uasset. The FileType.UNK file type is not supported in partition.
^T A1G!}2024-08-02 20:02:28.882519/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_Beveled_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_D.uasset. The FileType.UNK file type is not supported in partition.
   ; 	.;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               pW A1S!	2024-08-02 20:02:28.888554/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Tiles_Variation_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_Variation_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_Variation_M.uasset. The FileType.UNK file type is not supported in partition.
OV A1=!s2024-08-02 20:02:28.886397/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ground_Gravel_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Gravel_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Gravel_N.uasset. The FileType.UNK file type is not supported in partition.
   e 	Fe                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^Y A1G!}2024-08-02 20:02:28.892237/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Pebble_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_D.uasset. The FileType.UNK file type is not supported in partition.
7X A1-!cu2024-08-02 20:02:28.890505/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Burst_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Burst_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Burst_M.uasset. The FileType.UNK file type is not supported in partition.
   } 	7}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 7[ A1-!cu2024-08-02 20:02:28.898435/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Chair_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Chair_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Chair_M.uasset. The FileType.UNK file type is not supported in partition.
FZ A17!m2024-08-02 20:02:28.895370/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Panel_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Panel_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Panel_M.uasset. The FileType.UNK file type is not supported in partition.
   q 	+q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     7] A1-!cu2024-08-02 20:02:28.923070/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Shelf_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Shelf_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Shelf_N.uasset. The FileType.UNK file type is not supported in partition.
R\ A1?!u2024-08-02 20:02:28.901618/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Tiles_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_N.uasset. The FileType.UNK file type is not supported in partition.
   _ 	+_                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   I_ A19!o2024-08-02 20:02:28.928317/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Basalt_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Basalt_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Basalt_D.uasset. The FileType.UNK file type is not supported in partition.
R^ A1?!u2024-08-02 20:02:28.925329/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Aluminum_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Aluminum_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Aluminum_D.uasset. The FileType.UNK file type is not supported in partition.
   b 	(b                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      Ca A15!k}2024-08-02 20:02:28.937236/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Pine_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Pine_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Pine_N.uasset. The FileType.UNK file type is not supported in partition.
U` A1A!w	2024-08-02 20:02:28.931568/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Poured_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Poured_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Poured_N.uasset. The FileType.UNK file type is not supported in partition.
   t 	=t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Fc A17!m2024-08-02 20:02:28.945692/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_TableRound_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_TableRound_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_TableRound_N.uasset. The FileType.UNK file type is not supported in partition.
@b A13!i{2024-08-02 20:02:28.940748/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Dot_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Dot_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Dot_M.uasset. The FileType.UNK file type is not supported in partition.
   Y 	.Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Re A1?!u2024-08-02 20:02:28.961346/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_New_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_D.uasset. The FileType.UNK file type is not supported in partition.
Od A1=!s2024-08-02 20:02:28.948852/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Explosion_SubUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Explosion_SubUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Explosion_SubUV.uasset. The FileType.UNK file type is not supported in partition.
   q 	Fq                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Rg A1?!u2024-08-02 20:02:28.965330/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_Old_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Old_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Old_N.uasset. The FileType.UNK file type is not supported in partition.
7f A1-!cu2024-08-02 20:02:28.963523/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Water_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Water_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Water_N.uasset. The FileType.UNK file type is not supported in partition.
   M 	M                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 Ii A19!o2024-08-02 20:02:28.969521/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_ground_Moss_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_ground_Moss_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_ground_Moss_D.uasset. The FileType.UNK file type is not supported in partition.
dh A1K!2024-08-02 20:02:28.967971/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Marble_Polished_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Marble_Polished_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Marble_Polished_D.uasset. The FileType.UNK file type is not supported in partition.
   J 	J                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Ok A1=!s2024-08-02 20:02:28.994841/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Hex_Tile_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Hex_Tile_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Hex_Tile_M.uasset. The FileType.UNK file type is not supported in partition.
aj A1I!2024-08-02 20:02:28.981677/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Smooth_Granite_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Smooth_Granite_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Smooth_Granite_D.uasset. The FileType.UNK file type is not supported in partition.
   z 	Iz                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Lm A1;!q2024-08-02 20:02:28.998615/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ceramic_Tile_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ceramic_Tile_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ceramic_Tile_M.uasset. The FileType.UNK file type is not supported in partition.
4l A1+!as2024-08-02 20:02:28.996685/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Bush_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Bush_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Bush_N.uasset. The FileType.UNK file type is not supported in partition.
   Y 	Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Co A15!k}2024-08-02 20:02:29.022733/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Gradinet_01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Gradinet_01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Gradinet_01.uasset. The FileType.UNK file type is not supported in partition.
^n A1G!}2024-08-02 20:02:29.018294/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Pebble_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_N.uasset. The FileType.UNK file type is not supported in partition.
    	I                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          @q A13!i{2024-08-02 20:02:29.046109/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Spark_Core.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Spark_Core.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Spark_Core.uasset. The FileType.UNK file type is not supported in partition.
4p A1+!as2024-08-02 20:02:29.039636/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Door_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Door_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Door_M.uasset. The FileType.UNK file type is not supported in partition.
   t 	.t                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        7s A1-!cu2024-08-02 20:02:29.053703/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Frame_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Frame_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Frame_N.uasset. The FileType.UNK file type is not supported in partition.
Or A1=!s2024-08-02 20:02:29.050042/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ground_Gravel_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Gravel_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Gravel_D.uasset. The FileType.UNK file type is not supported in partition.
   > 	>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^u A1G!}2024-08-02 20:02:29.062675/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_Beveled_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_N.uasset. The FileType.UNK file type is not supported in partition.
^t A1G!}2024-08-02 20:02:29.056244/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Smooth_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_D.uasset. The FileType.UNK file type is not supported in partition.
   S 	+S                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       Uw A1A!w	2024-08-02 20:02:29.070184/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Cut_Stone_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Cut_Stone_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Cut_Stone_D.uasset. The FileType.UNK file type is not supported in partition.
Rv A1?!u2024-08-02 20:02:29.066496/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Grime_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Grime_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Grime_D.uasset. The FileType.UNK file type is not supported in partition.
   A 	"A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^y A1G!}2024-08-02 20:02:29.109563/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Smooth_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Smooth_M.uasset. The FileType.UNK file type is not supported in partition.
[x A1E!{2024-08-02 20:02:29.089055/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Floor_Walnut_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_N.uasset. The FileType.UNK file type is not supported in partition.
   Y 	(Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             L{ A1;!q2024-08-02 20:02:29.142192/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ground_Grass_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Grass_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Grass_D.uasset. The FileType.UNK file type is not supported in partition.
Uz A1A!w	2024-08-02 20:02:29.131543/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Panels_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Panels_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Panels_D.uasset. The FileType.UNK file type is not supported in partition.
   w 	Fw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           L} A1;!q2024-08-02 20:02:29.152484/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_MacroVariation.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_MacroVariation.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_MacroVariation.uasset. The FileType.UNK file type is not supported in partition.
7| A1-!cu2024-08-02 20:02:29.147240/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Chair_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Chair_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Chair_N.uasset. The FileType.UNK file type is not supported in partition.
   Y 	"Y                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             F A17!m2024-08-02 20:02:29.158582/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Gold_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Gold_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Gold_D.uasset. The FileType.UNK file type is not supported in partition.
[~ A1E!{2024-08-02 20:02:29.155565/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Rough_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Rough_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Rough_N.uasset. The FileType.UNK file type is not supported in partition.
   ` 	6`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    R A1?!u2024-08-02 20:02:29.195042/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Sandstone_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Sandstone_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Sandstone_N.uasset. The FileType.UNK file type is not supported in partition.
F  A17!m2024-08-02 20:02:29.163611/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Panel_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Panel_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Panel_N.uasset. The FileType.UNK file type is not supported in partition.
   W 	*W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           O A1=!s2024-08-02 20:02:29.217748/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Checker_Noise_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Checker_Noise_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Checker_Noise_M.uasset. The FileType.UNK file type is not supported in partition.
R A1?!u2024-08-02 20:02:29.198523/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_New_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_New_M.uasset. The FileType.UNK file type is not supported in partition.
   c 	6c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       O A1=!s2024-08-02 20:02:29.250161/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Dust_Particle_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Dust_Particle_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Dust_Particle_D.uasset. The FileType.UNK file type is not supported in partition.
F A17!m2024-08-02 20:02:29.244442/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Fire_Tiled_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Fire_Tiled_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Fire_Tiled_D.uasset. The FileType.UNK file type is not supported in partition.
   o 	3o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   @ A13!i{2024-08-02 20:02:29.268181/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_RockMesh_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_RockMesh_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_RockMesh_N.uasset. The FileType.UNK file type is not supported in partition.
I A19!o2024-08-02 20:02:29.258150/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Walnut_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Walnut_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Walnut_N.uasset. The FileType.UNK file type is not supported in partition.
   f 	$f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          :	 A1/!ew2024-08-02 20:02:29.358507/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Statue_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Statue_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Statue_N.uasset. The FileType.UNK file type is not supported in partition.
X A1C!y2024-08-02 20:02:29.350455/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Hewn_Stone_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_N.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	6~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  4 A1+!as2024-08-02 20:02:29.392927/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Lamp_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Lamp_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Lamp_N.uasset. The FileType.UNK file type is not supported in partition.
F
 A17!m2024-08-02 20:02:29.366372/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Slate_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Slate_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Slate_D.uasset. The FileType.UNK file type is not supported in partition.
   o 	3o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   @ A13!i{2024-08-02 20:02:29.408388/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Oak_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Oak_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Oak_D.uasset. The FileType.UNK file type is not supported in partition.
I A19!o2024-08-02 20:02:29.401775/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Single_Tile_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Single_Tile_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Single_Tile_N.uasset. The FileType.UNK file type is not supported in partition.
   i 	3i                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             F A17!m2024-08-02 20:02:29.450682/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Rust_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Rust_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Rust_N.uasset. The FileType.UNK file type is not supported in partition.
I A19!o2024-08-02 20:02:29.430771/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Steel_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Steel_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Steel_D.uasset. The FileType.UNK file type is not supported in partition.
   N 	0N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^ A1G!}2024-08-02 20:02:29.480538/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Pebble_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Pebble_M.uasset. The FileType.UNK file type is not supported in partition.
L A1;!q2024-08-02 20:02:29.458292/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Copper_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Copper_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Copper_D.uasset. The FileType.UNK file type is not supported in partition.
   ` 	*`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    F A17!m2024-08-02 20:02:29.490759/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Gold_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Gold_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Gold_N.uasset. The FileType.UNK file type is not supported in partition.
R A1?!u2024-08-02 20:02:29.486068/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Sandstone_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Sandstone_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Sandstone_D.uasset. The FileType.UNK file type is not supported in partition.
   Q 	0Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     [ A1E!{2024-08-02 20:02:29.511116/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_CobbleStone_Rough_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Rough_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_CobbleStone_Rough_D.uasset. The FileType.UNK file type is not supported in partition.
L A1;!q2024-08-02 20:02:29.505602/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ceramic_Tile_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ceramic_Tile_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ceramic_Tile_N.uasset. The FileType.UNK file type is not supported in partition.
   o 	3o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   @ A13!i{2024-08-02 20:02:29.518117/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Fire_SubUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Fire_SubUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Fire_SubUV.uasset. The FileType.UNK file type is not supported in partition.
I A19!o2024-08-02 20:02:29.515244/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Smoke_Tiled_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Smoke_Tiled_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Smoke_Tiled_D.uasset. The FileType.UNK file type is not supported in partition.
   W 	0W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           U A1A!w	2024-08-02 20:02:29.525644/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Panels_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Panels_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Panels_N.uasset. The FileType.UNK file type is not supported in partition.
L A1;!q2024-08-02 20:02:29.522560/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Ground_Grass_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Grass_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Ground_Grass_N.uasset. The FileType.UNK file type is not supported in partition.
   E 	E                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         U A1A!w	2024-08-02 20:02:29.570378/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Cut_Stone_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Cut_Stone_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Cut_Stone_N.uasset. The FileType.UNK file type is not supported in partition.
^ A1G!}2024-08-02 20:02:29.548786/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Clay_Beveled_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Clay_Beveled_M.uasset. The FileType.UNK file type is not supported in partition.
   f 	!f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          7 A1-!cu2024-08-02 20:02:29.579207/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Frame_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Frame_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Frame_M.uasset. The FileType.UNK file type is not supported in partition.
[ A1E!{2024-08-02 20:02:29.575485/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Floor_Walnut_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Floor_Walnut_D.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	H~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  F A17!m2024-08-02 20:02:29.594389/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_TableRound_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_TableRound_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_TableRound_M.uasset. The FileType.UNK file type is not supported in partition.
4 A1+!as2024-08-02 20:02:29.582287/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Door_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Door_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Door_N.uasset. The FileType.UNK file type is not supported in partition.
   i 	3i                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             F! A17!m2024-08-02 20:02:29.605549/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Rust_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Rust_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Rust_D.uasset. The FileType.UNK file type is not supported in partition.
I  A19!o2024-08-02 20:02:29.599673/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Metal_Steel_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Steel_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Metal_Steel_N.uasset. The FileType.UNK file type is not supported in partition.
   x 	<x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            @# A13!i{2024-08-02 20:02:29.634675/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Dot_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Dot_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Dot_N.uasset. The FileType.UNK file type is not supported in partition.
@" A13!i{2024-08-02 20:02:29.614690/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Oak_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Oak_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Oak_N.uasset. The FileType.UNK file type is not supported in partition.
   o 	Eo                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   R% A1?!u2024-08-02 20:02:29.655495/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Concrete_Tiles_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Concrete_Tiles_M.uasset. The FileType.UNK file type is not supported in partition.
7$ A1-!cu2024-08-02 20:02:29.639470/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Shelf_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Shelf_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Shelf_M.uasset. The FileType.UNK file type is not supported in partition.
   c 	6c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       O' A1=!s2024-08-02 20:02:29.664629/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Tech_Hex_Tile_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Hex_Tile_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Tech_Hex_Tile_N.uasset. The FileType.UNK file type is not supported in partition.
F& A17!m2024-08-02 20:02:29.660152/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Rock_Slate_N.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Slate_N.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Rock_Slate_N.uasset. The FileType.UNK file type is not supported in partition.
   W 	$W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I) A19!o2024-08-02 20:02:29.670671/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Wood_Walnut_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Walnut_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Wood_Walnut_D.uasset. The FileType.UNK file type is not supported in partition.
X( A1C!y2024-08-02 20:02:29.668068/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Brick_Hewn_Stone_D.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_D.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Brick_Hewn_Stone_D.uasset. The FileType.UNK file type is not supported in partition.
    	E                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 4+ A3)!as2024-08-02 20:02:29.682379/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Glass.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Glass.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Glass.uasset. The FileType.UNK file type is not supported in partition.
7* A1-!cu2024-08-02 20:02:29.673459/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/TexturesT_Water_M.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Water_M.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Textures/T_Water_M.uasset. The FileType.UNK file type is not supported in partition.
   E 	*E                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         a- A3G!2024-08-02 20:02:29.698573/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Brushed_Nickel.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Brushed_Nickel.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Brushed_Nickel.uasset. The FileType.UNK file type is not supported in partition.
R, A3=!u2024-08-02 20:02:29.689845/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Brick_Cut_Stone.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Cut_Stone.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Cut_Stone.uasset. The FileType.UNK file type is not supported in partition.
   c 	$c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       =/ A3/!gy2024-08-02 20:02:29.708954/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Wood_Oak.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Oak.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Oak.uasset. The FileType.UNK file type is not supported in partition.
X. A3A!y2024-08-02 20:02:29.701777/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_CobbleStone_Rough.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Rough.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Rough.uasset. The FileType.UNK file type is not supported in partition.
   K 	K                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               L1 A39!q2024-08-02 20:02:29.725915/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_AssetPlatform.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_AssetPlatform.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_AssetPlatform.uasset. The FileType.UNK file type is not supported in partition.
a0 A3G!2024-08-02 20:02:29.717339/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Rock_Marble_Polished.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Marble_Polished.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Marble_Polished.uasset. The FileType.UNK file type is not supported in partition.
   E 	<E                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s3 A3S!2024-08-02 20:02:29.746997/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Wood_Floor_Walnut_Polished.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.uasset. The FileType.UNK file type is not supported in partition.
@2 A31!i{2024-08-02 20:02:29.733293/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Wood_Pine.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Pine.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Pine.uasset. The FileType.UNK file type is not supported in partition.
   o 	6o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   C5 A33!k}2024-08-02 20:02:29.765571/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Rust.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Rust.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Rust.uasset. The FileType.UNK file type is not supported in partition.
F4 A35!m2024-08-02 20:02:29.755523/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Rock_Basalt.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Basalt.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Basalt.uasset. The FileType.UNK file type is not supported in partition.
   N 	9N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  g7 A3K!2024-08-02 20:02:29.812533/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Wood_Floor_Walnut_Worn.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Floor_Walnut_Worn.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Floor_Walnut_Worn.uasset. The FileType.UNK file type is not supported in partition.
C6 A33!k}2024-08-02 20:02:29.784913/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Water_Lake.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Water_Lake.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Water_Lake.uasset. The FileType.UNK file type is not supported in partition.
   c 	-c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       F9 A35!m2024-08-02 20:02:29.828821/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Water_Ocean.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Water_Ocean.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Water_Ocean.uasset. The FileType.UNK file type is not supported in partition.
O8 A3;!s2024-08-02 20:02:29.820243/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Brick_Clay_Old.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_Old.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_Old.uasset. The FileType.UNK file type is not supported in partition.
   Z 	!Z                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              C; A33!k}2024-08-02 20:02:29.854536/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Rock_Slate.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Slate.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Slate.uasset. The FileType.UNK file type is not supported in partition.
[: A3C!{2024-08-02 20:02:29.845332/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Brick_Clay_Beveled.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_Beveled.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_Beveled.uasset. The FileType.UNK file type is not supported in partition.
   c 	*c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       C= A33!k}2024-08-02 20:02:29.882499/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Basic_Wall.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Basic_Wall.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Basic_Wall.uasset. The FileType.UNK file type is not supported in partition.
R< A3=!u2024-08-02 20:02:29.866339/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Concrete_Panels.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Panels.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Panels.uasset. The FileType.UNK file type is not supported in partition.
   c 	*c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       C? A33!k}2024-08-02 20:02:29.910415/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Gold.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Gold.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Gold.uasset. The FileType.UNK file type is not supported in partition.
R> A3=!u2024-08-02 20:02:29.888623/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Concrete_Poured.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Poured.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Poured.uasset. The FileType.UNK file type is not supported in partition.
   T 	!T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        IA A37!o2024-08-02 20:02:29.920994/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Ground_Grass.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Grass.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Grass.uasset. The FileType.UNK file type is not supported in partition.
[@ A3C!{2024-08-02 20:02:29.916474/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_CobbleStone_Smooth.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Smooth.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Smooth.uasset. The FileType.UNK file type is not supported in partition.
   f 	-f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          CC A33!k}2024-08-02 20:02:29.932956/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Tech_Panel.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Panel.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Panel.uasset. The FileType.UNK file type is not supported in partition.
OB A3;!s2024-08-02 20:02:29.929327/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Brick_Clay_New.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_New.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Clay_New.uasset. The FileType.UNK file type is not supported in partition.
   i 	6i                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             IE A37!o2024-08-02 20:02:29.979589/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Copper.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Copper.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Copper.uasset. The FileType.UNK file type is not supported in partition.
FD A35!m2024-08-02 20:02:29.971606/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Steel.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Steel.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Steel.uasset. The FileType.UNK file type is not supported in partition.
   6 	6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^G A3E!}2024-08-02 20:02:29.993808/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Tech_Hex_Tile_Pulse.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Hex_Tile_Pulse.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Hex_Tile_Pulse.uasset. The FileType.UNK file type is not supported in partition.
dF A3I!2024-08-02 20:02:29.988095/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Burnished_Steel.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Burnished_Steel.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Burnished_Steel.uasset. The FileType.UNK file type is not supported in partition.
   K 	'K                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               XI A3A!y2024-08-02 20:02:30.004939/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_ColorGrid_LowSpec.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_ColorGrid_LowSpec.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_ColorGrid_LowSpec.uasset. The FileType.UNK file type is not supported in partition.
UH A3?!w	2024-08-02 20:02:29.998846/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Brick_Hewn_Stone.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Hewn_Stone.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Brick_Hewn_Stone.uasset. The FileType.UNK file type is not supported in partition.
   Q 	6Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     aK A3G!2024-08-02 20:02:30.016768/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Ceramic_Tile_Checker.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ceramic_Tile_Checker.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ceramic_Tile_Checker.uasset. The FileType.UNK file type is not supported in partition.
FJ A35!m2024-08-02 20:02:30.011211/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Wood_Walnut.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Walnut.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Wood_Walnut.uasset. The FileType.UNK file type is not supported in partition.
   c 	-c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       FM A35!m2024-08-02 20:02:30.037671/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Basic_Floor.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Basic_Floor.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Basic_Floor.uasset. The FileType.UNK file type is not supported in partition.
OL A3;!s2024-08-02 20:02:30.027155/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Concrete_Grime.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Grime.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Grime.uasset. The FileType.UNK file type is not supported in partition.
   ] 	0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 OO A3;!s2024-08-02 20:02:30.054701/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Concrete_Tiles.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Tiles.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Concrete_Tiles.uasset. The FileType.UNK file type is not supported in partition.
LN A39!q2024-08-02 20:02:30.042315/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Ground_Gravel.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Gravel.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Gravel.uasset. The FileType.UNK file type is not supported in partition.
   T 	'T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        OQ A3;!s2024-08-02 20:02:30.068042/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Rock_Sandstone.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Sandstone.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Rock_Sandstone.uasset. The FileType.UNK file type is not supported in partition.
UP A3?!w	2024-08-02 20:02:30.059897/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Tech_Checker_Dot.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Checker_Dot.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Checker_Dot.uasset. The FileType.UNK file type is not supported in partition.
   i 	6i                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             IS A37!o2024-08-02 20:02:30.085739/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Metal_Chrome.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Chrome.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Metal_Chrome.uasset. The FileType.UNK file type is not supported in partition.
FR A35!m2024-08-02 20:02:30.071721/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Ground_Moss.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Moss.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Ground_Moss.uasset. The FileType.UNK file type is not supported in partition.
   Q 	!Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     LU A39!q2024-08-02 20:02:30.129853/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_Tech_Hex_Tile.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Hex_Tile.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_Tech_Hex_Tile.uasset. The FileType.UNK file type is not supported in partition.
[T A3C!{2024-08-02 20:02:30.096656/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/MaterialsM_CobbleStone_Pebble.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Pebble.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Materials/M_CobbleStone_Pebble.uasset. The FileType.UNK file type is not supported in partition.
   $ 	$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        pW A5O!	2024-08-02 20:02:30.167570/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_Effect_Explosion.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Explosion.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Explosion.uasset. The FileType.UNK file type is not supported in partition.
dV A5G!2024-08-02 20:02:30.159895/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_Effect_Smoke.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Smoke.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Smoke.uasset. The FileType.UNK file type is not supported in partition.
   N 	0N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^Y A5C!}2024-08-02 20:02:30.183840/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_WallSconce.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_WallSconce.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_WallSconce.uasset. The FileType.UNK file type is not supported in partition.
LX A57!q2024-08-02 20:02:30.177010/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBP_LightStudio.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/BP_LightStudio.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/BP_LightStudio.uasset. The FileType.UNK file type is not supported in partition.
   0 	0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    d[ A5G!2024-08-02 20:02:30.209350/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_CeilingLight.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_CeilingLight.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_CeilingLight.uasset. The FileType.UNK file type is not supported in partition.
dZ A5G!2024-08-02 20:02:30.189709/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_Effect_Steam.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Steam.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Steam.uasset. The FileType.UNK file type is not supported in partition.
   0 	0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    g] A5I!2024-08-02 20:02:30.221007/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_Effect_Sparks.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Sparks.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Sparks.uasset. The FileType.UNK file type is not supported in partition.
a\ A5E!2024-08-02 20:02:30.215927/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/BlueprintsBlueprint_Effect_Fire.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Fire.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Blueprint_Effect_Fire.uasset. The FileType.UNK file type is not supported in partition.
   6 	6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          R_ AC-!u2024-08-02 20:02:30.264594/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsSM_Arrows.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/SM_Arrows.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/SM_Arrows.uasset. The FileType.UNK file type is not supported in partition.
p^ ACA!	2024-08-02 20:02:30.233236/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsM_LightStage_Arrows.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Arrows.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Arrows.uasset. The FileType.UNK file type is not supported in partition.
   ÷ ú÷                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           a ACK!%2024-08-02 20:02:30.284699/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsM_LightStage_Skybox_HDRI.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_HDRI.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_HDRI.uasset. The FileType.UNK file type is not supported in partition.
` ACM!'2024-08-02 20:02:30.270011/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsM_LightStage_Skybox_Black.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_Black.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_Black.uasset. The FileType.UNK file type is not supported in partition.
   * 	*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              gc AC;!2024-08-02 20:02:30.302674/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsFogBrightnessLUT.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/FogBrightnessLUT.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/FogBrightnessLUT.uasset. The FileType.UNK file type is not supported in partition.
gb AC;!2024-08-02 20:02:30.291997/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsSunlightColorLUT.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/SunlightColorLUT.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/SunlightColorLUT.uasset. The FileType.UNK file type is not supported in partition.
   * ÷*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Ie AC'!o2024-08-02 20:02:30.335064/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsSkybox.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/Skybox.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/Skybox.uasset. The FileType.UNK file type is not supported in partition.
d ACO!)2024-08-02 20:02:30.318769/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/AssetsM_LightStage_Skybox_Master.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_Master.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Blueprints/Assets/M_LightStage_Skybox_Master.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	?~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  =g A+7!gy2024-08-02 20:02:30.345049/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsMaterialSphere.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialSphere.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialSphere.uasset. The FileType.UNK file type is not supported in partition.
=f A+7!gy2024-08-02 20:02:30.340414/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_PillarFrame.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_PillarFrame.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_PillarFrame.uasset. The FileType.UNK file type is not supported in partition.
    	?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     :i A+5!ew2024-08-02 20:02:30.355033/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_TableRound.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_TableRound.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_TableRound.uasset. The FileType.UNK file type is not supported in partition.
=h A+7!gy2024-08-02 20:02:30.350767/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_CornerFrame.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_CornerFrame.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_CornerFrame.uasset. The FileType.UNK file type is not supported in partition.
   ¥ 	T¥                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         +k A++![m2024-08-02 20:02:30.362252/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Chair.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Chair.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Chair.uasset. The FileType.UNK file type is not supported in partition.
(j A+)!Yk2024-08-02 20:02:30.358186/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Door.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Door.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Door.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	-~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  +m A++![m2024-08-02 20:02:30.367864/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Couch.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Couch.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Couch.uasset. The FileType.UNK file type is not supported in partition.
Ol A+C!s2024-08-02 20:02:30.365214/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_MatPreviewMesh_02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_MatPreviewMesh_02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_MatPreviewMesh_02.uasset. The FileType.UNK file type is not supported in partition.
    	?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    +o A++![m2024-08-02 20:02:30.381648/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Shelf.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Shelf.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Shelf.uasset. The FileType.UNK file type is not supported in partition.
=n A+7!gy2024-08-02 20:02:30.370526/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_GlassWindow.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_GlassWindow.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_GlassWindow.uasset. The FileType.UNK file type is not supported in partition.
    	T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             7q A+3!cu2024-08-02 20:02:30.396952/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_DoorFrame.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_DoorFrame.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_DoorFrame.uasset. The FileType.UNK file type is not supported in partition.
(p A+)!Yk2024-08-02 20:02:30.393559/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Rock.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Rock.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Rock.uasset. The FileType.UNK file type is not supported in partition.
    	N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Fs A+=!m2024-08-02 20:02:30.402530/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_PillarFrame300.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_PillarFrame300.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_PillarFrame300.uasset. The FileType.UNK file type is not supported in partition.
.r A+-!]o2024-08-02 20:02:30.400188/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Statue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Statue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Statue.uasset. The FileType.UNK file type is not supported in partition.
    	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              .u A+-!]o2024-08-02 20:02:30.412743/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Stairs.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Stairs.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Stairs.uasset. The FileType.UNK file type is not supported in partition.
@t A+9!i{2024-08-02 20:02:30.405368/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Lamp_Ceiling.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Lamp_Ceiling.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Lamp_Ceiling.uasset. The FileType.UNK file type is not supported in partition.
    	?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       (w A+)!Yk2024-08-02 20:02:30.424954/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Bush.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Bush.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Bush.uasset. The FileType.UNK file type is not supported in partition.
=v A+7!gy2024-08-02 20:02:30.422216/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_WindowFrame.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_WindowFrame.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_WindowFrame.uasset. The FileType.UNK file type is not supported in partition.
   l 	El                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Uy A?3!w	2024-08-02 20:02:30.471722/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_TableRound.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_TableRound.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_TableRound.uasset. The FileType.UNK file type is not supported in partition.
7x A+3!cu2024-08-02 20:02:30.434603/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/PropsSM_Lamp_Wall.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Lamp_Wall.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/SM_Lamp_Wall.uasset. The FileType.UNK file type is not supported in partition.
   o 	9o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   F{ A?)!m2024-08-02 20:02:30.502073/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Chair.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Chair.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Chair.uasset. The FileType.UNK file type is not supported in partition.
Cz A?'!k}2024-08-02 20:02:30.490410/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Rock.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Rock.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Rock.uasset. The FileType.UNK file type is not supported in partition.
   W 	$W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I} A?+!o2024-08-02 20:02:30.532127/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Statue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Statue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Statue.uasset. The FileType.UNK file type is not supported in partition.
X| A?5!y2024-08-02 20:02:30.514600/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_StatueGlass.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_StatueGlass.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_StatueGlass.uasset. The FileType.UNK file type is not supported in partition.
   T 	T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        C A?'!k}2024-08-02 20:02:30.552195/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Bush.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Bush.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Bush.uasset. The FileType.UNK file type is not supported in partition.
a~ A?;!2024-08-02 20:02:30.538181/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_MaterialSphere.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_MaterialSphere.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_MaterialSphere.uasset. The FileType.UNK file type is not supported in partition.
   B 		B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      C A?'!k}2024-08-02 20:02:30.561877/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Door.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Door.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Door.uasset. The FileType.UNK file type is not supported in partition.
s  A?G!2024-08-02 20:02:30.558750/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_MaterialSphere_Plain.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_MaterialSphere_Plain.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_MaterialSphere_Plain.uasset. The FileType.UNK file type is not supported in partition.
   o 	6o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   C A?'!k}2024-08-02 20:02:30.570090/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Lamp.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Lamp.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Lamp.uasset. The FileType.UNK file type is not supported in partition.
F A?)!m2024-08-02 20:02:30.567380/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Shelf.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Shelf.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Shelf.uasset. The FileType.UNK file type is not supported in partition.
   u 	6u                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         = A-5!gy2024-08-02 20:02:30.583721/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Wedge_A.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Wedge_A.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Wedge_A.uasset. The FileType.UNK file type is not supported in partition.
F A?)!m2024-08-02 20:02:30.575067/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/MaterialsM_Frame.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Frame.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Props/Materials/M_Frame.uasset. The FileType.UNK file type is not supported in partition.
   u 	3u                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         : A-3!ew2024-08-02 20:02:30.599348/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Sphere.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Sphere.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Sphere.uasset. The FileType.UNK file type is not supported in partition.
I A-=!o2024-08-02 20:02:30.592542/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_WideCapsule.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_WideCapsule.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_WideCapsule.uasset. The FileType.UNK file type is not supported in partition.
    	H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        @	 A-7!i{2024-08-02 20:02:30.605875/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Cylinder.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cylinder.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cylinder.uasset. The FileType.UNK file type is not supported in partition.
4 A-/!as2024-08-02 20:02:30.602827/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Tube.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Tube.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Tube.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	6~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  4 A-/!as2024-08-02 20:02:30.615356/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Trim.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim.uasset. The FileType.UNK file type is not supported in partition.
F
 A-;!m2024-08-02 20:02:30.609244/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_TriPyramid.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_TriPyramid.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_TriPyramid.uasset. The FileType.UNK file type is not supported in partition.
   x 	Ex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            I A-=!o2024-08-02 20:02:30.621688/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Trim_90_Out.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim_90_Out.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim_90_Out.uasset. The FileType.UNK file type is not supported in partition.
7 A-1!cu2024-08-02 20:02:30.619152/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Plane.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Plane.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Plane.uasset. The FileType.UNK file type is not supported in partition.
   { 	?{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               @ A-7!i{2024-08-02 20:02:30.627906/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Pipe_180.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe_180.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe_180.uasset. The FileType.UNK file type is not supported in partition.
= A-5!gy2024-08-02 20:02:30.626198/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Pipe_90.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe_90.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe_90.uasset. The FileType.UNK file type is not supported in partition.
    	H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    4 A-/!as2024-08-02 20:02:30.639296/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Cube.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cube.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cube.uasset. The FileType.UNK file type is not supported in partition.
4 A-/!as2024-08-02 20:02:30.633993/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Pipe.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Pipe.uasset. The FileType.UNK file type is not supported in partition.
    	H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           = A-5!gy2024-08-02 20:02:30.644428/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Wedge_B.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Wedge_B.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Wedge_B.uasset. The FileType.UNK file type is not supported in partition.
4 A-/!as2024-08-02 20:02:30.642718/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Cone.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cone.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Cone.uasset. The FileType.UNK file type is not supported in partition.
   ` 	-`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    I A-=!o2024-08-02 20:02:30.648454/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_QuadPyramid.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_QuadPyramid.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_QuadPyramid.uasset. The FileType.UNK file type is not supported in partition.
O A-A!s2024-08-02 20:02:30.646658/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_NarrowCapsule.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_NarrowCapsule.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_NarrowCapsule.uasset. The FileType.UNK file type is not supported in partition.
   { 	6{                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               7 A-1!cu2024-08-02 20:02:30.654650/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Torus.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Torus.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Torus.uasset. The FileType.UNK file type is not supported in partition.
F A-;!m2024-08-02 20:02:30.652618/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ShapesShape_Trim_90_In.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim_90_In.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Shapes/Shape_Trim_90_In.uasset. The FileType.UNK file type is not supported in partition.
    	K                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    7 A3+!cu2024-08-02 20:02:30.670865/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Sparks.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Sparks.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Sparks.uasset. The FileType.UNK file type is not supported in partition.
1 A3'!_q2024-08-02 20:02:30.667984/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Fire.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Fire.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Fire.uasset. The FileType.UNK file type is not supported in partition.
   o 	3o                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   @ A31!i{2024-08-02 20:02:30.677762/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Explosion.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Explosion.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Explosion.uasset. The FileType.UNK file type is not supported in partition.
I A37!o2024-08-02 20:02:30.675756/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Ambient_Dust.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Ambient_Dust.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Ambient_Dust.uasset. The FileType.UNK file type is not supported in partition.
    	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        4 A3)!as2024-08-02 20:02:30.691254/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Smoke.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Smoke.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Smoke.uasset. The FileType.UNK file type is not supported in partition.
@ A31!i{2024-08-02 20:02:30.688513/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ParticlesP_Steam_Lit.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Steam_Lit.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/P_Steam_Lit.uasset. The FileType.UNK file type is not supported in partition.
   $ 	$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        d AG5!2024-08-02 20:02:30.696163/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_smoke_subUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_smoke_subUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_smoke_subUV.uasset. The FileType.UNK file type is not supported in partition.
p AG=!	2024-08-02 20:02:30.694274/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Heat_Distortion.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Heat_Distortion.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Heat_Distortion.uasset. The FileType.UNK file type is not supported in partition.
   - 	-                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 a! AG3!2024-08-02 20:02:30.708311/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Fire_SubUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Fire_SubUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Fire_SubUV.uasset. The FileType.UNK file type is not supported in partition.
j  AG9!2024-08-02 20:02:30.705219/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Dust_Particle.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Dust_Particle.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Dust_Particle.uasset. The FileType.UNK file type is not supported in partition.
   $ 	$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        d# AG5!2024-08-02 20:02:30.730217/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_radial_ramp.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_radial_ramp.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_radial_ramp.uasset. The FileType.UNK file type is not supported in partition.
p" AG=!	2024-08-02 20:02:30.714503/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Radial_Gradient.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Radial_Gradient.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Radial_Gradient.uasset. The FileType.UNK file type is not supported in partition.
   - 	!-                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 p% AG=!	2024-08-02 20:02:30.759045/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_explosion_subUV.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_explosion_subUV.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_explosion_subUV.uasset. The FileType.UNK file type is not supported in partition.
[$ AG/!{2024-08-02 20:02:30.750966/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materialsm_flare_01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/m_flare_01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/m_flare_01.uasset. The FileType.UNK file type is not supported in partition.
   T 	*T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        R' AG)!u2024-08-02 20:02:30.771892/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Burst.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Burst.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Burst.uasset. The FileType.UNK file type is not supported in partition.
R& AG)!u2024-08-02 20:02:30.761810/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/MaterialsM_Spark.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Spark.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Particles/Materials/M_Spark.uasset. The FileType.UNK file type is not supported in partition.
   Q 	!Q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     L) A93!q2024-08-02 20:02:30.782531/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_400x400.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x400.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x400.uasset. The FileType.UNK file type is not supported in partition.
[( A9=!{2024-08-02 20:02:30.774791/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_Door_400x400.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Door_400x400.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Door_400x400.uasset. The FileType.UNK file type is not supported in partition.
   T 	0T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        X+ A9;!y2024-08-02 20:02:30.808513/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureSM_AssetPlatform.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/SM_AssetPlatform.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/SM_AssetPlatform.uasset. The FileType.UNK file type is not supported in partition.
L* A93!q2024-08-02 20:02:30.789912/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_500x500.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_500x500.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_500x500.uasset. The FileType.UNK file type is not supported in partition.
   N 	-N                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [- A9=!{2024-08-02 20:02:30.832200/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_Door_400x300.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Door_400x300.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Door_400x300.uasset. The FileType.UNK file type is not supported in partition.
O, A95!s2024-08-02 20:02:30.821693/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitecturePillar_50x500.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Pillar_50x500.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Pillar_50x500.uasset. The FileType.UNK file type is not supported in partition.
   ` 	0`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    L/ A93!q2024-08-02 20:02:30.858683/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_400x300.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x300.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x300.uasset. The FileType.UNK file type is not supported in partition.
L. A93!q2024-08-02 20:02:30.845606/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_400x200.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x200.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_400x200.uasset. The FileType.UNK file type is not supported in partition.
   H 	H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            O1 A95!s2024-08-02 20:02:30.866713/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureFloor_400x400.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Floor_400x400.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Floor_400x400.uasset. The FileType.UNK file type is not supported in partition.
a0 A9A!2024-08-02 20:02:30.863445/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_Window_400x400.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Window_400x400.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Window_400x400.uasset. The FileType.UNK file type is not supported in partition.
   c 	c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       43 A+1!as2024-08-02 20:02:30.888258/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioExplosion02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion02.uasset. The FileType.UNK file type is not supported in partition.
a2 A9A!2024-08-02 20:02:30.874453/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/ArchitectureWall_Window_400x300.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Window_400x300.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Architecture/Wall_Window_400x300.uasset. The FileType.UNK file type is not supported in partition.
    	B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     =5 A+7!gy2024-08-02 20:02:30.907388/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Wind06.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Wind06.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Wind06.uasset. The FileType.UNK file type is not supported in partition.
:4 A+5!ew2024-08-02 20:02:30.897252/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioExplosion_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion_Cue.uasset. The FileType.UNK file type is not supported in partition.
    	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     77 A+3!cu2024-08-02 20:02:30.917567/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioCollapse_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse_Cue.uasset. The FileType.UNK file type is not supported in partition.
@6 A+9!i{2024-08-02 20:02:30.912822/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Birds01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Birds01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Birds01.uasset. The FileType.UNK file type is not supported in partition.
    	B                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          (9 A+)!Yk2024-08-02 20:02:30.968218/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioSmoke01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Smoke01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Smoke01.uasset. The FileType.UNK file type is not supported in partition.
:8 A+5!ew2024-08-02 20:02:30.951351/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioFire_Sparks01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire_Sparks01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire_Sparks01.uasset. The FileType.UNK file type is not supported in partition.
   r 	Kr                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      U; A+G!w	2024-08-02 20:02:30.989688/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Background_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Background_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Background_Cue.uasset. The FileType.UNK file type is not supported in partition.
1: A+/!_q2024-08-02 20:02:30.982819/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioCollapse01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse01.uasset. The FileType.UNK file type is not supported in partition.
    	T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                4= A+1!as2024-08-02 20:02:31.010160/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioLight02_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light02_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light02_Cue.uasset. The FileType.UNK file type is not supported in partition.
(< A+)!Yk2024-08-02 20:02:31.003384/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioLight02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light02.uasset. The FileType.UNK file type is not supported in partition.
   ~ 	H~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  F? A+=!m2024-08-02 20:02:31.027462/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioFire_Sparks01_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire_Sparks01_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire_Sparks01_Cue.uasset. The FileType.UNK file type is not supported in partition.
4> A+1!as2024-08-02 20:02:31.021489/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioSmoke01_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Smoke01_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Smoke01_Cue.uasset. The FileType.UNK file type is not supported in partition.
   ¢ 	W¢                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      1A A+/!_q2024-08-02 20:02:31.038656/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioFire01_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire01_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire01_Cue.uasset. The FileType.UNK file type is not supported in partition.
%@ A+'!Wi2024-08-02 20:02:31.032122/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioFire01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Fire01.uasset. The FileType.UNK file type is not supported in partition.
    	H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    4C A+1!as2024-08-02 20:02:31.050220/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioSteam01_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Steam01_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Steam01_Cue.uasset. The FileType.UNK file type is not supported in partition.
4B A+1!as2024-08-02 20:02:31.043685/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioLight01_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light01_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light01_Cue.uasset. The FileType.UNK file type is not supported in partition.
    	?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       (E A+)!Yk2024-08-02 20:02:31.080441/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioSteam01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Steam01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Steam01.uasset. The FileType.UNK file type is not supported in partition.
=D A+7!gy2024-08-02 20:02:31.053772/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Wind05.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Wind05.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Wind05.uasset. The FileType.UNK file type is not supported in partition.
    	<                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        4G A+1!as2024-08-02 20:02:31.103384/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioExplosion01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Explosion01.uasset. The FileType.UNK file type is not supported in partition.
@F A+9!i{2024-08-02 20:02:31.086986/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Music01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Music01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Music01.uasset. The FileType.UNK file type is not supported in partition.
    	T                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   1I A+/!_q2024-08-02 20:02:31.133204/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioCollapse02.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse02.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Collapse02.uasset. The FileType.UNK file type is not supported in partition.
(H A+)!Yk2024-08-02 20:02:31.110704/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioLight01.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light01.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Light01.uasset. The FileType.UNK file type is not supported in partition.
   ¢ 	6¢                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      K A;!K]2024-08-02 20:02:31.155009/Users/antonio/Documents/Unreal Projects/MyProject/IntermediateCachedAssetRegistry.binbinValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/CachedAssetRegistry.bin. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/CachedAssetRegistry.bin. The FileType.UNK file type is not supported in partition.
FJ A+=!m2024-08-02 20:02:31.140592/Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/AudioStarter_Music_Cue.uassetuassetValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Music_Cue.uasset. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Content/StarterContent/Audio/Starter_Music_Cue.uasset. The FileType.UNK file type is not supported in partition.
   f 	]f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          sM AGA!2024-08-02 20:02:31.176882/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsUnrealInsightsSettings.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/UnrealInsightsSettings.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/UnrealInsightsSettings.ini. The FileType.UNK file type is not supported in partition.
L A')!Ug2024-08-02 20:02:31.164965/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ReimportCache3688439234.binbinValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ReimportCache/3688439234.bin. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ReimportCache/3688439234.bin. The FileType.UNK file type is not supported in partition.
    	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         O AGK!)2024-08-02 20:02:31.190140/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsLocalizationServiceSettings.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/LocalizationServiceSettings.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/LocalizationServiceSettings.ini. The FileType.UNK file type is not supported in partition.
aN AG5!2024-08-02 20:02:31.185325/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsGameUserSettings.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/GameUserSettings.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/GameUserSettings.ini. The FileType.UNK file type is not supported in partition.
    	$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               Q AGM!+2024-08-02 20:02:31.202992/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsRemoteControlProtocolWidgets.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/RemoteControlProtocolWidgets.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/RemoteControlProtocolWidgets.ini. The FileType.UNK file type is not supported in partition.
XP AG/!{2024-08-02 20:02:31.194530/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsVirtualCamera.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/VirtualCamera.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/VirtualCamera.ini. The FileType.UNK file type is not supported in partition.
   f 	9f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          OS AG)!u2024-08-02 20:02:31.209221/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsEncryption.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Encryption.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Encryption.ini. The FileType.UNK file type is not supported in partition.
CR AG!!m2024-08-02 20:02:31.206196/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsCrypto.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Crypto.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Crypto.ini. The FileType.UNK file type is not supported in partition.
   H 	9H                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            mU AG=!	2024-08-02 20:02:31.229983/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_MACES3AutogenShaderHeaders.ushushValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_MACES3/AutogenShaderHeaders.ush. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_MACES3/AutogenShaderHeaders.ush. The FileType.UNK file type is not supported in partition.
CT AG!!m2024-08-02 20:02:31.225938/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigsEngine.iniiniValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Engine.ini. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Config/CoalescedSourceConfigs/Engine.ini. The FileType.UNK file type is not supported in partition.
    	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           W Ai1!12024-08-02 20:02:31.243510/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/0WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/0/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/0/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
dV AA=!2024-08-02 20:02:31.235219/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_SM5AutogenShaderHeaders.ushushValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_SM5/AutogenShaderHeaders.ush. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/ShaderAutogen/SF_METAL_SM5/AutogenShaderHeaders.ush. The FileType.UNK file type is not supported in partition.
   Þ ïÞ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Y Ai1!12024-08-02 20:02:31.280774/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/4WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/4/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/4/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
X Ai1!12024-08-02 20:02:31.271963/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/1WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/1/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/1/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
   Þ ïÞ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [ Ai1!12024-08-02 20:02:31.331062/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/2WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/2/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/2/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
Z Ai1!12024-08-02 20:02:31.326856/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/3WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/3/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/3/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
    E ïS E                                                       
^ AI#!q2024-08-02 20:02:40.039682/Users/antonio/Documents/BlurbMiami.blurbblurbValueErrorInvalid file /Users/antonio/Documents/Blurb/Miami.blurb. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Blurb/Miami.blurb. The FileType.UNK file type is not supported in partition.
¥] A_A%IÃ2024-08-02 20:02:35.368153/Users/antonio/Documents/Digital EditionsThe Trump Indictments.epubepubRuntimeErrorPandoc died with exitcode "1" during conversion: pandoc: Cannot decode byte '\x93': Data.Text.Internal.Encoding.streamDecodeUtf8With: Invalid UTF-8 stream




Current version of pandoc: 2.19.2
Make sure you have the right version installed in your system. Please, follow the pandoc installation instructions in README.md to install the right version.Traceback (most recent call last):
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/file_conversion.py", line 15, in convert_file_to_text
      ±\ Ai1!12024-08-02 20:02:31.343234/Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/5WorkerInputOnly.ininValueErrorInvalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/5/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Unreal Projects/MyProject/Intermediate/Shaders/tmp/54F3C368DC4E46F9540F72B5421F8CED/5/WorkerInputOnly.in. The FileType.UNK file type is not supported in partition.
     text = pypandoc.convert_file(filename, target_format, format=source_format)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/pypandoc/__init__.py", line 200, in convert_file
    return _convert_input(discovered_source_files, format, 'path', to, extra_args=extra_args,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/pypandoc/__init__.py", line 467, in _convert_input
    raise RuntimeError(
RuntimeError: Pandoc died with exitcode "1" during conversion: pandoc: Cannot decode byte '\x93': Data.Text.Internal.Encoding.streamDecodeUtf8With: Invalid UTF-8 stream


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/epub.py", line 42, in _get_elements
    return partition_epub(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/documents/elements.py", line 587, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/filetype.py", line 618, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/filetype.py", line 582, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/chunking/dispatch.py", line 74, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/epub.py", line 55, in partition_epub
    elements = convert_and_partition_html(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/html.py", line 219, in convert_and_partition_html
    html_text = convert_file_to_html_text(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/file_conversion.py", line 68, in convert_file_to_html_text
    html_text = convert_file_to_text(
                ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/utils.py", line 245, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/file_conversion.py", line 43, in convert_file_to_text
    raise RuntimeError(msg)
RuntimeError: Pandoc died with exitcode "1" during conversion: pandoc: Cannot decode byte '\x93': Data.Text.Internal.Encoding.streamDecodeUtf8With: Invalid UTF-8 stream




Current version of pandoc: 2.19.2
Make sure you have the right version installed in your system. Please, follow the pandoc installation instructions in README.md to install the right version.
   ¼ 	æ¼                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                &` A]#!2024-08-02 20:02:40.044260/Users/antonio/Documents/RDC ConnectionsDefault.rdprdpValueErrorInvalid file /Users/antonio/Documents/RDC Connections/Default.rdp. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/RDC Connections/Default.rdp. The FileType.UNK file type is not supported in partition.
_ AI+!y2024-08-02 20:02:40.042175/Users/antonio/Documents/BlurbPatagonia.blurbblurbValueErrorInvalid file /Users/antonio/Documents/Blurb/Patagonia.blurb. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Blurb/Patagonia.blurb. The FileType.UNK file type is not supported in partition.
    	É                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ;b AW3!!2024-08-02 20:02:49.262860/Users/antonio/Documents/FreeFileSyncRaid2EVO1.ffs_batchffs_batchValueErrorInvalid file /Users/antonio/Documents/FreeFileSync/Raid2EVO1.ffs_batch. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/FreeFileSync/Raid2EVO1.ffs_batch. The FileType.UNK file type is not supported in partition.
3a AW/!2024-08-02 20:02:49.260605/Users/antonio/Documents/FreeFileSyncEVO12Raid.ffs_guiffs_guiValueErrorInvalid file /Users/antonio/Documents/FreeFileSync/EVO12Raid.ffs_gui. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/FreeFileSync/EVO12Raid.ffs_gui. The FileType.UNK file type is not supported in partition.
    	Ç                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          -d AW+!2024-08-02 20:02:49.265923/Users/antonio/Documents/FreeFileSyncLibrary.ffs_guiffs_guiValueErrorInvalid file /Users/antonio/Documents/FreeFileSync/Library.ffs_gui. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/FreeFileSync/Library.ffs_gui. The FileType.UNK file type is not supported in partition.
5c AW/!2024-08-02 20:02:49.264446/Users/antonio/Documents/FreeFileSyncLibrary.ffs_batchffs_batchValueErrorInvalid file /Users/antonio/Documents/FreeFileSync/Library.ffs_batch. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/FreeFileSync/Library.ffs_batch. The FileType.UNK file type is not supported in partition.
   N 	ÉN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  wf AcS!;M2024-08-02 20:02:49.272679/Users/antonio/Documents/H&R Block BusinessAntcarRealtyLLC 2017 Tax Return.atxatxValueErrorInvalid file /Users/antonio/Documents/H&R Block Business/AntcarRealtyLLC 2017 Tax Return.atx. The FileType.ZIP file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/H&R Block Business/AntcarRealtyLLC 2017 Tax Return.atx. The FileType.ZIP file type is not supported in partition.
3e AW/!2024-08-02 20:02:49.267309/Users/antonio/Documents/FreeFileSyncRaid2EVO1.ffs_guiffs_guiValueErrorInvalid file /Users/antonio/Documents/FreeFileSync/Raid2EVO1.ffs_gui. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/FreeFileSync/Raid2EVO1.ffs_gui. The FileType.UNK file type is not supported in partition.
   m 	»m                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 Jh As%!/2024-08-02 20:02:53.345444/Users/antonio/Documents/Adobe/Pixel Bender/samplespixelate.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/pixelate.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/pixelate.pbk. The FileType.UNK file type is not supported in partition.
Ag As!)2024-08-02 20:02:53.343319/Users/antonio/Documents/Adobe/Pixel Bender/samplessepia.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/sepia.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/sepia.pbk. The FileType.UNK file type is not supported in partition.
   F 	¯F                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ej As7!/A2024-08-02 20:02:53.348132/Users/antonio/Documents/Adobe/Pixel Bender/samplesalphaFromMaxColor.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/alphaFromMaxColor.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/alphaFromMaxColor.pbk. The FileType.UNK file type is not supported in partition.
Mi As'!12024-08-02 20:02:53.346957/Users/antonio/Documents/Adobe/Pixel Bender/samplescrossfade.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/crossfade.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/crossfade.pbk. The FileType.UNK file type is not supported in partition.
   ^ 	£^                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Al As!)2024-08-02 20:02:53.350782/Users/antonio/Documents/Adobe/Pixel Bender/samplestwirl.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/twirl.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/twirl.pbk. The FileType.UNK file type is not supported in partition.
Yk As/!'92024-08-02 20:02:53.349395/Users/antonio/Documents/Adobe/Pixel Bender/samplessimpleBoxBlur.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/simpleBoxBlur.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/simpleBoxBlur.pbk. The FileType.UNK file type is not supported in partition.
    	¯                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               n A-!K]2024-08-02 20:02:53.360244/Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatiblebasicBoxBlur.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatible/basicBoxBlur.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatible/basicBoxBlur.pbk. The FileType.UNK file type is not supported in partition.
Mm As'!12024-08-02 20:02:53.352984/Users/antonio/Documents/Adobe/Pixel Bender/samplesinvertRGB.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/invertRGB.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/invertRGB.pbk. The FileType.UNK file type is not supported in partition.
   ð 	lð                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    xp A5!;M2024-08-02 20:02:53.365480/Users/antonio/Documents/Adobe/Pixel Bender/samples/graphsCombineTwoInputs.pbgpbgValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/CombineTwoInputs.pbg. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/CombineTwoInputs.pbg. The FileType.UNK file type is not supported in partition.
o A-!K]2024-08-02 20:02:53.361562/Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatiblecheckerboard.pbkpbkValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatible/checkerboard.pbk. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/notFlashCompatible/checkerboard.pbk. The FileType.UNK file type is not supported in partition.
   / 	/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ir A+!1C2024-08-02 20:02:53.368810/Users/antonio/Documents/Adobe/Pixel Bender/samples/graphssepia-twirl.pbgpbgValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/sepia-twirl.pbg. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/sepia-twirl.pbg. The FileType.UNK file type is not supported in partition.
`q A%!+=2024-08-02 20:02:53.367560/Users/antonio/Documents/Adobe/Pixel Bender/samples/graphspixelate.pbgpbgValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/pixelate.pbg. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/pixelate.pbg. The FileType.UNK file type is not supported in partition.
   	 	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ]s A#!);2024-08-02 20:02:53.373758/Users/antonio/Documents/Adobe/Pixel Bender/samples/graphsinFocus.pbgpbgValueErrorInvalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/inFocus.pbg. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Pixel Bender/samples/graphs/inFocus.pbg. The FileType.UNK file type is not supported in partition.
   Û Û                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 !t As/)+±=2024-08-02 20:02:53.376913/Users/antonio/Documents/Adobe/Lumetri/9.0/settingsUser_Settings.xmlxmlXMLSyntaxErrorStart tag expected, '<' not found, line 3, column 1 (User_Settings.xml, line 3)Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 372, in partition
    elements = partition_xml(
               ^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/documents/elements.py", line 587, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/filetype.py", line 618, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/file_utils/filetype.py", line 582, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/chunking/dispatch.py", line 74, in wrapper
    elements = func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/xml.py", line 168, in partition_xml
    for leaf_element in leaf_elements:
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/xml.py", line 63, in _get_leaf_elements
    for event, element in element_iterator:
  File "src/lxml/iterparse.pxi", line 208, in lxml.etree.iterparse.__next__
  File "src/lxml/iterparse.pxi", line 193, in lxml.etree.iterparse.__next__
  File "src/lxml/iterparse.pxi", line 224, in lxml.etree.iterparse._read_more_events
  File "src/lxml/parser.pxi", line 1486, in lxml.etree._FeedParser.close
  File "src/lxml/parser.pxi", line 624, in lxml.etree._ParserContext._handleParseResult
  File "src/lxml/parser.pxi", line 633, in lxml.etree._ParserContext._handleParseResultDoc
  File "src/lxml/parser.pxi", line 743, in lxml.etree._handleParseResult
  File "src/lxml/parser.pxi", line 672, in lxml.etree._raiseParseError
  File "/Users/antonio/Documents/Adobe/Lumetri/9.0/settings/User_Settings.xml", line 3
lxml.etree.XMLSyntaxError: Start tag expected, '<' not found, line 3, column 1
   É 	fÉ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             v AA!Se2024-08-02 20:02:53.386855/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonioSharedView Column SettingsValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/SharedView Column Settings. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/SharedView Column Settings. The FileType.UNK file type is not supported in partition.
u A;!M_2024-08-02 20:02:53.384404/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonioInstalled Guides.guidesguidesValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Installed Guides.guides. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Installed Guides.guides. The FileType.UNK file type is not supported in partition.
   · 	Q·                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           x A?!Qc2024-08-02 20:02:53.389608/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonioAdobe Premiere Rush PrefsValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Adobe Premiere Rush Prefs. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Adobe Premiere Rush Prefs. The FileType.UNK file type is not supported in partition.
+w AM!_q2024-08-02 20:02:53.388340/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonioMedia Browser Provider ExceptionValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Media Browser Provider Exception. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Media Browser Provider Exception. The FileType.UNK file type is not supported in partition.
   ¨ 	{¨                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Oz A?1!u2024-08-02 20:02:53.392194/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/Overlay PresetsCustom Overlay.olpolpValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/Overlay Presets/Custom Overlay.olp. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/Overlay Presets/Custom Overlay.olp. The FileType.UNK file type is not supported in partition.
y A1!CU2024-08-02 20:02:53.390802/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonioRecent DirectoriesValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Recent Directories. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Recent Directories. The FileType.UNK file type is not supported in partition.
   Õ ìÕ                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | AY=)!-2024-08-02 20:02:53.396947/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
{ AY;)!+2024-08-02 20:02:53.395058/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/music/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
   Û òÛ                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ~ A[;)!-2024-08-02 20:02:53.399380/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog/(Config).essentialsound. The FileType.UNK file type is not supported in partition.

} AU;)!'2024-08-02 20:02:53.398090/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/sfx(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/sfx/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/sfx/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
   Ì æÌ                                                                                                                                                                                                                                                                                                                                                                                                                                                                  A];)!/2024-08-02 20:02:53.403875/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
 A[=)!/2024-08-02 20:02:53.400444/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/dialog/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
   Æ ãÆ                                                                                                                                                                                                                                                                                                                                                                                                                                                           A_;)!12024-08-02 20:02:53.408303/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/ambience(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/ambience/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/ambience/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
 A]=)!12024-08-02 20:02:53.405395/Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.0/Profile-antonio/Settings/EssentialSound/Default/generic/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
   É 	fÉ                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              AA!Se2024-08-02 20:02:53.413190/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-AntonioSharedView Column SettingsValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/SharedView Column Settings. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/SharedView Column Settings. The FileType.UNK file type is not supported in partition.
 A;!M_2024-08-02 20:02:53.412058/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-AntonioInstalled Guides.guidesguidesValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Installed Guides.guides. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Installed Guides.guides. The FileType.UNK file type is not supported in partition.
   · 	Q·                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            A?!Qc2024-08-02 20:02:53.415850/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-AntonioAdobe Premiere Rush PrefsValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Adobe Premiere Rush Prefs. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Adobe Premiere Rush Prefs. The FileType.UNK file type is not supported in partition.
+ AM!_q2024-08-02 20:02:53.414618/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-AntonioMedia Browser Provider ExceptionValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Media Browser Provider Exception. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Media Browser Provider Exception. The FileType.UNK file type is not supported in partition.
   ¨ 	{¨                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            O A?1!u2024-08-02 20:02:53.421131/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/Overlay PresetsCustom Overlay.olpolpValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/Overlay Presets/Custom Overlay.olp. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/Overlay Presets/Custom Overlay.olp. The FileType.UNK file type is not supported in partition.
 A1!CU2024-08-02 20:02:53.418470/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-AntonioRecent DirectoriesValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Recent Directories. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Recent Directories. The FileType.UNK file type is not supported in partition.
   Õ ìÕ                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
 AY=)!-2024-08-02 20:02:53.423780/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
	 AY;)!+2024-08-02 20:02:53.422628/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
    ¡                                                                                                                                                                                                                                                                                                                                                                                                       
 AU;)!'2024-08-02 20:02:53.426161/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/sfx(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/sfx/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/sfx/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
[ AYm)!K]2024-08-02 20:02:53.424854/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music(RushDefaultAutoLoudnessDisabled).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/music/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.
   Ï éÏ                                                                                                                                                                                                                                                                                                                                                                                                                                                                    A[=)!/2024-08-02 20:02:53.431075/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
 A[;)!-2024-08-02 20:02:53.429817/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
                                                                                                                                                                                                                                                                                                                                                                                             A];)!/2024-08-02 20:02:53.435329/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
^ A[m)!M_2024-08-02 20:02:53.433558/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog(RushDefaultAutoLoudnessDisabled).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/dialog/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.
   ~ ã~                                                                                                                                                                                                                                                                                                                                                                                  a A]m)!Oa2024-08-02 20:02:53.438061/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic(RushDefaultAutoLoudnessDisabled).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(RushDefaultAutoLoudnessDisabled).essentialsound. The FileType.UNK file type is not supported in partition.
 A]=)!12024-08-02 20:02:53.436378/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic(Default).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(Default).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/generic/(Default).essentialsound. The FileType.UNK file type is not supported in partition.
   \ ã\                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 AyE!CU2024-08-02 20:03:04.649587/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915153141612.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915153141612.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915153141612.MOV. The FileType.UNK file type is not supported in partition.
 A_;)!12024-08-02 20:02:53.443387/Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/ambience(Config).essentialsoundessentialsoundValueErrorInvalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/ambience/(Config).essentialsound. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/Adobe/Premiere Rush/1.5/Profile-Antonio/Settings/EssentialSound/Default/ambience/(Config).essentialsound. The FileType.UNK file type is not supported in partition.
   " 	©"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       AyE!CU2024-08-02 20:03:04.657403/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915152900859.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915152900859.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915152900859.MOV. The FileType.UNK file type is not supported in partition.
S Ay%!#52024-08-02 20:03:04.652981/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0928.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0928.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0928.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S Ay%!#52024-08-02 20:03:04.661689/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0939.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0939.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0939.MOV. The FileType.UNK file type is not supported in partition.
S Ay%!#52024-08-02 20:03:04.659422/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0929.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0929.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0929.MOV. The FileType.UNK file type is not supported in partition.
   " 	y"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S Ay%!#52024-08-02 20:03:04.669715/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0938.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0938.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0938.MOV. The FileType.UNK file type is not supported in partition.
 AyE!CU2024-08-02 20:03:04.667615/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915151140366.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915151140366.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915151140366.MOV. The FileType.UNK file type is not supported in partition.
   ò 	yò                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       AyE!CU2024-08-02 20:03:04.677601/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915150946547.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915150946547.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915150946547.MOV. The FileType.UNK file type is not supported in partition.
 AyE!CU2024-08-02 20:03:04.673499/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915150915928.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915150915928.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915150915928.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S Ay%!#52024-08-02 20:03:04.683978/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0922.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0922.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0922.MOV. The FileType.UNK file type is not supported in partition.
S Ay%!#52024-08-02 20:03:04.681847/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0940.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0940.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0940.MOV. The FileType.UNK file type is not supported in partition.
   " 	©"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        AyE!CU2024-08-02 20:03:04.692641/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915151021152.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915151021152.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915151021152.MOV. The FileType.UNK file type is not supported in partition.
S Ay%!#52024-08-02 20:03:04.688129/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0936.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0936.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0936.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S" Ay%!#52024-08-02 20:03:04.699114/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0935.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0935.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0935.MOV. The FileType.UNK file type is not supported in partition.
S! Ay%!#52024-08-02 20:03:04.696010/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0937.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0937.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0937.MOV. The FileType.UNK file type is not supported in partition.
   " 	y"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S$ Ay%!#52024-08-02 20:03:04.702971/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0934.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0934.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0934.MOV. The FileType.UNK file type is not supported in partition.
# AyE!CU2024-08-02 20:03:04.701030/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughMoment-20200915152541314.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915152541314.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/Moment-20200915152541314.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S& Ay%!#52024-08-02 20:03:04.707868/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0925.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0925.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0925.MOV. The FileType.UNK file type is not supported in partition.
S% Ay%!#52024-08-02 20:03:04.704970/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0930.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0930.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0930.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S( Ay%!#52024-08-02 20:03:04.714215/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0927.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0927.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0927.MOV. The FileType.UNK file type is not supported in partition.
S' Ay%!#52024-08-02 20:03:04.712496/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0931.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0931.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0931.MOV. The FileType.UNK file type is not supported in partition.
   R 	©R                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      S* Ay%!#52024-08-02 20:03:04.719447/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0932.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0932.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0932.MOV. The FileType.UNK file type is not supported in partition.
S) Ay%!#52024-08-02 20:03:04.716325/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0933.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0933.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0933.MOV. The FileType.UNK file type is not supported in partition.
   	© 	©                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               S+ Ay%!#52024-08-02 20:03:04.723819/Users/antonio/Documents/11211SW82PL/2020/Walk-ThroughIMG_0926.MOVMOVValueErrorInvalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0926.MOV. The FileType.UNK file type is not supported in partition.Traceback (most recent call last):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_directory.py", line 72, in load_directory_lazy
    for doc in load_document_lazy(file_path, **kwargs):
  File "/Users/antonio/Desktop/DataScience/MyCode/langchain-document_db/examples/document_loaders/../../document_loaders/load_document.py", line 81, in load_document_lazy
    for doc in loader_method():
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 89, in lazy_load
    elements = self._get_elements()
               ^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/langchain_community/document_loaders/unstructured.py", line 181, in _get_elements
    return partition(filename=self.file_path, **self.unstructured_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/antonio/miniconda3/envs/myenv/lib/python3.12/site-packages/unstructured/partition/auto.py", line 545, in partition
    raise ValueError(f"{msg}. The {filetype} file type is not supported in partition.")
ValueError: Invalid file /Users/antonio/Documents/11211SW82PL/2020/Walk-Through/IMG_0926.MOV. The FileType.UNK file type is not supported in partition.
