The Thesis-MSC LaTeX template can be used for Instituto Superior Técnico Master of Science thesis of DEI (or other departments), as it follows the regulations published by the Scientific Council of IST.

To start using, just open and modify your data in the following files:

1. Thesis-MSc-Preamble_commands.tex
Select the ‘main’ language of your thesis in the line that loads the package ‘babel’:
line.22
	\usepackage[main=english,portuguese]{babel}	
	or 
	\usepackage[english,main=portuguese]{babel}
	
The thesis document will then auto-magically modify all the adequate keywords, titles, captions, etc. according to the selected language.

Later, you may modify parameters or include other packages if deemed necessary.

To deliver a PDF version without annotations/todonotes, Track Changes, you need to modify the options as follows:

For TODO Notes, the following lines:
line.121/122
    % use the line with the 'disable' option for the TODO Notes
    \usepackage[textwidth=2cm, textsize=small, disable]{todonotes}
 
For Track Changes, the following lines: 
line.130/131
    % use the line with the 'final' option for the Track Changes
    \usepackage[authormarkup=superscript,authormarkuptext=id,markup=underlined,ulem={ULforem,normalbf},final]{changes}




2. Thesis-MSc-Front_Cover.tex
In this document insert your data (your full name, thesis titles, supervisors names, date, degree, etc.):

	\title{}
	\subtitle{}
	\author{}
	\degree{}
	\supervisor{}
	\othersupervisor{}
	\date{}

For Tracking changes, you can also replace the id of the collaborators, in the lines:

    \definechangesauthor[color=colorname]{id}

For a draft thesis (before examination) modify the following command to the value ‘false’:

	\finalthesis{false}

If it is for the final version (approved after examination) then modify the command to the value ‘true’:

	\finalthesis{true}

and include the full names of all the members of the Examination Committee (except the Supervisor):

	\chairperson{}
	\vogalone{}
	\vogaltwo{}
	\vogalthree{}

3. The content of your thesis will be written in the documents in folder ‘Chapters’
Do not modify the ‘header’ in all those documents, except the name of the document as it contains compilation directives.
 
If you need to add more Chapters, just create new .tex documents and add the header.
If you do not need that many Chapters in your thesis, just comment the lines in the main document:

  	‘Thesis-MSc-Main_Document.tex’

For example, if you do not need a 6th Chapter, comment the following lines:

	%Chapter 6
	\input{./Chapters/Thesis-MSc-Chapter_6.tex}
	% If Printing on DOUBLE SIDED pages, the second page should be white.
	% Otherwise, comment the following command:
	\cleardoublepage

4. Thesis-MSc-Bibliography.bib
This file is the default Bibliography database. 
If you are using Mendeley.com as Reference Manager, you can link the .bib file to your account in that swervice in order to keep the data automatically synchronized.
For those purposes, replace that file with your own file, and if the name is different, just modify the corresponding line in in the main document ‘Thesis-MSc-Main_Document.tex’:

	\bibliography{./Thesis-MSc-Bibliography}

#####################

Enjoy

Rui Santos Cruz