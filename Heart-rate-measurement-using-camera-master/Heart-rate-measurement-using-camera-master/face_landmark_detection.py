<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<!-- saved from url=(0047)http://dlib.net/face_landmark_detection.py.html -->
<html class="gr__dlib_net"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title></title>
  
  <style type="text/css">
td.linenos { background-color: #f0f0f0; padding-right: 10px; }
span.lineno { background-color: #f0f0f0; padding: 0 5px 0 5px; }
pre { line-height: 125%; }
body .hll { background-color: #ffffcc }
body  { background: #ffffff; }
body .c { color: #008000 } /* Comment */
body .err { border: 1px solid #FF0000 } /* Error */
body .k { color: #0000ff } /* Keyword */
body .ch { color: #008000 } /* Comment.Hashbang */
body .cm { color: #008000 } /* Comment.Multiline */
body .cp { color: #0000ff } /* Comment.Preproc */
body .cpf { color: #008000 } /* Comment.PreprocFile */
body .c1 { color: #008000 } /* Comment.Single */
body .cs { color: #008000 } /* Comment.Special */
body .ge { font-style: italic } /* Generic.Emph */
body .gh { font-weight: bold } /* Generic.Heading */
body .gp { font-weight: bold } /* Generic.Prompt */
body .gs { font-weight: bold } /* Generic.Strong */
body .gu { font-weight: bold } /* Generic.Subheading */
body .kc { color: #0000ff } /* Keyword.Constant */
body .kd { color: #0000ff } /* Keyword.Declaration */
body .kn { color: #0000ff } /* Keyword.Namespace */
body .kp { color: #0000ff } /* Keyword.Pseudo */
body .kr { color: #0000ff } /* Keyword.Reserved */
body .kt { color: #2b91af } /* Keyword.Type */
body .s { color: #a31515 } /* Literal.String */
body .nc { color: #2b91af } /* Name.Class */
body .ow { color: #0000ff } /* Operator.Word */
body .sa { color: #a31515 } /* Literal.String.Affix */
body .sb { color: #a31515 } /* Literal.String.Backtick */
body .sc { color: #a31515 } /* Literal.String.Char */
body .dl { color: #a31515 } /* Literal.String.Delimiter */
body .sd { color: #a31515 } /* Literal.String.Doc */
body .s2 { color: #a31515 } /* Literal.String.Double */
body .se { color: #a31515 } /* Literal.String.Escape */
body .sh { color: #a31515 } /* Literal.String.Heredoc */
body .si { color: #a31515 } /* Literal.String.Interpol */
body .sx { color: #a31515 } /* Literal.String.Other */
body .sr { color: #a31515 } /* Literal.String.Regex */
body .s1 { color: #a31515 } /* Literal.String.Single */
body .ss { color: #a31515 } /* Literal.String.Symbol */

  </style>
</head>
<body data-gr-c-s-loaded="true">
<h2></h2>

<div class="highlight"><pre><span></span><span class="ch">#!/usr/bin/python</span>
<span class="c1"># The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt</span>
<span class="c1">#</span>
<span class="c1">#   This example program shows how to find frontal human faces in an image and</span>
<span class="c1">#   estimate their pose.  The pose takes the form of 68 landmarks.  These are</span>
<span class="c1">#   points on the face such as the corners of the mouth, along the eyebrows, on</span>
<span class="c1">#   the eyes, and so forth.</span>
<span class="c1">#</span>
<span class="c1">#   The face detector we use is made using the classic Histogram of Oriented</span>
<span class="c1">#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,</span>
<span class="c1">#   and sliding window detection scheme.  The pose estimator was created by</span>
<span class="c1">#   using dlib's implementation of the paper:</span>
<span class="c1">#      One Millisecond Face Alignment with an Ensemble of Regression Trees by</span>
<span class="c1">#      Vahid Kazemi and Josephine Sullivan, CVPR 2014</span>
<span class="c1">#   and was trained on the iBUG 300-W face landmark dataset (see</span>
<span class="c1">#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  </span>
<span class="c1">#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. </span>
<span class="c1">#      300 faces In-the-wild challenge: Database and results. </span>
<span class="c1">#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.</span>
<span class="c1">#   You can get the trained model file from:</span>
<span class="c1">#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.</span>
<span class="c1">#   Note that the license for the iBUG 300-W dataset excludes commercial use.</span>
<span class="c1">#   So you should contact Imperial College London to find out if it's OK for</span>
<span class="c1">#   you to use this model file in a commercial product.</span>
<span class="c1">#</span>
<span class="c1">#</span>
<span class="c1">#   Also, note that you can train your own models using dlib's machine learning</span>
<span class="c1">#   tools. See <a href="http://dlib.net/train_shape_predictor.py.html">train_shape_predictor.py</a> to see an example.</span>
<span class="c1">#</span>
<span class="c1">#</span>
<span class="c1"># COMPILING/INSTALLING THE DLIB PYTHON INTERFACE</span>
<span class="c1">#   You can install dlib using the command:</span>
<span class="c1">#       pip install dlib</span>
<span class="c1">#</span>
<span class="c1">#   Alternatively, if you want to compile dlib yourself then go into the dlib</span>
<span class="c1">#   root folder and run:</span>
<span class="c1">#       python setup.py install</span>
<span class="c1">#</span>
<span class="c1">#   Compiling dlib should work on any operating system so long as you have</span>
<span class="c1">#   CMake installed.  On Ubuntu, this can be done easily by running the</span>
<span class="c1">#   command:</span>
<span class="c1">#       sudo apt-get install cmake</span>
<span class="c1">#</span>
<span class="c1">#   Also note that this example requires Numpy which can be installed</span>
<span class="c1">#   via the command:</span>
<span class="c1">#       pip install numpy</span>

<span class="kn">import</span> <span class="nn">sys</span>
<span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">dlib</span>
<span class="kn">import</span> <span class="nn">glob</span>

<span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">sys</span><span class="o">.</span><span class="n">argv</span><span class="p">)</span> <span class="o">!=</span> <span class="mi">3</span><span class="p">:</span>
    <span class="k">print</span><span class="p">(</span>
        <span class="s2">"Give the path to the trained shape predictor model as the first "</span>
        <span class="s2">"argument and then the directory containing the facial images.</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"For example, if you are in the python_examples folder then "</span>
        <span class="s2">"execute this program by running:</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"    ./<a href="http://dlib.net/face_landmark_detection.py.html">face_landmark_detection.py</a> shape_predictor_68_face_landmarks.dat ../examples/faces</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"You can download a trained facial shape predictor from:</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"</span><span class="p">)</span>
    <span class="nb">exit</span><span class="p">()</span>

<span class="n">predictor_path</span> <span class="o">=</span> <span class="n">sys</span><span class="o">.</span><span class="n">argv</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
<span class="n">faces_folder_path</span> <span class="o">=</span> <span class="n">sys</span><span class="o">.</span><span class="n">argv</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span>

<span class="n">detector</span> <span class="o">=</span> <span class="n">dlib</span><span class="o">.</span><span class="n">get_frontal_face_detector</span><span class="p">()</span>
<span class="n">predictor</span> <span class="o">=</span> <span class="n">dlib</span><span class="o">.</span><span class="n">shape_predictor</span><span class="p">(</span><span class="n">predictor_path</span><span class="p">)</span>
<span class="n">win</span> <span class="o">=</span> <span class="n">dlib</span><span class="o">.</span><span class="n">image_window</span><span class="p">()</span>

<span class="k">for</span> <span class="n">f</span> <span class="ow">in</span> <span class="n">glob</span><span class="o">.</span><span class="n">glob</span><span class="p">(</span><span class="n">os</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">faces_folder_path</span><span class="p">,</span> <span class="s2">"*.jpg"</span><span class="p">)):</span>
    <span class="k">print</span><span class="p">(</span><span class="s2">"Processing file: {}"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">f</span><span class="p">))</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">dlib</span><span class="o">.</span><span class="n">load_rgb_image</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>

    <span class="n">win</span><span class="o">.</span><span class="n">clear_overlay</span><span class="p">()</span>
    <span class="n">win</span><span class="o">.</span><span class="n">set_image</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>

    <span class="c1"># Ask the detector to find the bounding boxes of each face. The 1 in the</span>
    <span class="c1"># second argument indicates that we should upsample the image 1 time. This</span>
    <span class="c1"># will make everything bigger and allow us to detect more faces.</span>
    <span class="n">dets</span> <span class="o">=</span> <span class="n">detector</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span>
    <span class="k">print</span><span class="p">(</span><span class="s2">"Number of faces detected: {}"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">dets</span><span class="p">)))</span>
    <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">d</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">dets</span><span class="p">):</span>
        <span class="k">print</span><span class="p">(</span><span class="s2">"Detection {}: Left: {} Top: {} Right: {} Bottom: {}"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span>
            <span class="n">k</span><span class="p">,</span> <span class="n">d</span><span class="o">.</span><span class="n">left</span><span class="p">(),</span> <span class="n">d</span><span class="o">.</span><span class="n">top</span><span class="p">(),</span> <span class="n">d</span><span class="o">.</span><span class="n">right</span><span class="p">(),</span> <span class="n">d</span><span class="o">.</span><span class="n">bottom</span><span class="p">()))</span>
        <span class="c1"># Get the landmarks/parts for the face in box d.</span>
        <span class="n">shape</span> <span class="o">=</span> <span class="n">predictor</span><span class="p">(</span><span class="n">img</span><span class="p">,</span> <span class="n">d</span><span class="p">)</span>
        <span class="k">print</span><span class="p">(</span><span class="s2">"Part 0: {}, Part 1: {} ..."</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">shape</span><span class="o">.</span><span class="n">part</span><span class="p">(</span><span class="mi">0</span><span class="p">),</span>
                                                  <span class="n">shape</span><span class="o">.</span><span class="n">part</span><span class="p">(</span><span class="mi">1</span><span class="p">)))</span>
        <span class="c1"># Draw the face landmarks on the screen.</span>
        <span class="n">win</span><span class="o">.</span><span class="n">add_overlay</span><span class="p">(</span><span class="n">shape</span><span class="p">)</span>

    <span class="n">win</span><span class="o">.</span><span class="n">add_overlay</span><span class="p">(</span><span class="n">dets</span><span class="p">)</span>
    <span class="n">dlib</span><span class="o">.</span><span class="n">hit_enter_to_continue</span><span class="p">()</span>
</pre></div>


</body><div><div class="gr_-editor gr-iframe-first-load" style="display: none;"><div class="gr_-editor_back"></div><iframe class="gr_-ifr gr-_dialog-content" src="./face_landmark_detection_files/saved_resource.html"></iframe></div></div><grammarly-card><div></div></grammarly-card><span class="gr__tooltip"><span class="gr__tooltip-content"></span><i class="gr__tooltip-logo"></i><span class="gr__triangle"></span></span></html>