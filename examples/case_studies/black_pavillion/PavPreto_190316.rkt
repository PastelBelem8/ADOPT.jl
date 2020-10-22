#lang racket
;(require rosetta/rhino)
(require rosetta/lighting-simulation)
;(require "../lighting-simulation.rkt")
(require "materials.rkt")
(provide (all-defined-out))


(define (itera-quadrangulos f ptss)
  (for/list 
    ((pts0 ptss)
     (pts1 (cdr ptss)))
    (for/list
      ((p0 pts0)
       (p1 pts1)
       (p2 (cdr pts1))
       (p3 (cdr pts0)))
      (f p0 p1 p2 p3))))

;----------------------------------------------------- SLABS -------------------------------------------------------;
;the surface for the analysis, as well as, the position of the sensors is defined in this function
(define (slabs p c l pd d c-a l-a)
  (parameterize ((default-slab-family
                   (slab-family-element
                    (default-slab-family)
                    #:thickness 0.25)))
    (slab (reverse
           (list p
                 (+y p c)
                 (+xy p l c)
                 (+x p l))))
    (let ((p1 (+z p (+ 0.25 pd))))
      (parameterize (#;(analyze-surfaces #t))
        (slab (reverse
               (list p1
                     (+y p1 c)
                     (+xy p1 l c)
                     (+xy p1 l (+ c-a d))
                     (+xy p1 (- l l-a) (+ c-a d))
                     (+xy p1 (- l l-a) d)
                     (+xy p1 l d)
                     (+x p1 l)))))
      ;SURFACE
      (let ((p1 (+xyz p1 0.59 0.525 0.001)))
        (itera-quadrangulos
         (lambda (p0 p1 p2 p3)
           (add-radiance-polygon!
            (list p0 p1 p2 p3)))
         ;NODES
         (map-division (lambda (x y)
                         (+xy p1 x y))
                       0 (- l 0.59 0.3) 7
                       0 (- d 0.525) 11))))))

;Node separation 0.6m - 23 32
;Node separation 1.0m - 14 22
;Node separation 1.5m - 9 15
;Node separation 2.0m - 7 11

      

;--------------------------------------------------- PASSADIÇO -----------------------------------------------------;
(define (passadico p length width height thickness)
  (let ((p1 (+z p height))
        (p2 (+x p width))
        (p3 (+xz p width height)))
    (add-radiance-shape!
     (shape-material
      (thicken (extrusion (line p1 p p2 p3) (vy length p))
               thickness)
      generic-metal)))) ;o thicken está a fazer offset para fora - ter em consideração na coordenada Z


;---------------------------------------------------- COLUMNS ------------------------------------------------------;
;;FAMILIES
(define column-type1 (column-family-element
                     (default-column-family)
                     #:width 0.82
                     #:depth 0.82))

(define column-type2 (column-family-element
                     (default-column-family)
                     #:width 0.71
                     #:depth 0.36))

;;COLUMNS
(define (columns-type1 p h c l nc nl)
  (parameterize ((default-column-family column-type1)
                 (default-level-to-level-height h)
                 #;(current-level (level 0))
                 (material/column concrete))
    (union
     (map-division (lambda (x) (column (+x p x)))
                0 l (- nl 1) #f)
     (map-division (lambda (y) (column (+xy p l y)))
                0 c (- nc 1) #f)
     (map-division (lambda (x) (column (+xy p x c)))
                l 0 (- nl 1)))))

(define (columns-type2 p h c n)
  (parameterize ((default-column-family column-type2)
                 (default-level-to-level-height h)
                 #;(current-level (level 0))
                 (material/column concrete))
    (union
     (map column
          (rest 
           (map-division (lambda (y) (+y p y))
                         0 c (- n 1) #f))))))


;----------------------------------------------------- WALLS -------------------------------------------------------;
;;FAMILIES
(define wall-muro (wall-family-element
                        (default-wall-family)
                        #:thickness 0.41))

(define wall-exterior (wall-family-element
                        (default-wall-family)
                        #:thickness 0.35))

(define wall-framing (wall-family-element
                      (default-wall-family)
                      #:thickness 0.30))

(define wall-type1 (wall-family-element
                        (default-wall-family)
                        #:thickness 0.20))

(define wall-type2 (wall-family-element
                        (default-wall-family)
                        #:thickness 0.05))

;;MUROS
(define (muros p c l c1-1 c1-2 c2 h1 h2)
  (union
   (parameterize ((default-wall-family wall-muro)
                  (default-level-to-level-height h1)
                  #;(current-level (level 0))
                  (material/wall outside-facade-35))
     (wall (+xy p (- (/ l 2) 1.6) (- 0.35))
           (+xy p (- (/ l 2) 1.6) (- (+ c1-1 0.35))))
     (wall (+xy p (- (/ l 2) 1.6) (- (+ c1-1 0.35)))
           (+xy p (+ l 0.205) (- (+ c1-1 0.35))))
     (wall (+xy p (+ l 0.205) (- (+ c1-1 0.35)))
           (+xy p (+ l 0.205) (- (+ c1-1 c1-2 0.35)))))
   (parameterize ((default-wall-family
                    (wall-family-element
                     (default-wall-family)
                     #:thickness 0.30))
                  (default-level-to-level-height h2)
                  #;(current-level (level 0))
                  (material/wall outside-facade-35))
     (wall (+xy p (+ l 0.41) (+ c 0.15))
           (+xy p (+ l 0.41 c2) (+ c 0.15))))))

;;EXTERIOR
(define (exterior-walls p c l h)
  (parameterize ((default-wall-family wall-exterior)
                 (default-level-to-level-height h)
                 #;(current-level (level 0))
                 (material/wall outside-facade-35))
    (union
     (wall p (+x p (/ l 2)))
     (parameterize ((default-window-family
                      (window-family-element
                       (default-window-family)
                       #:width 37.74
                       #:height 0.70)))
       (window
        (wall p (+y p c))
        (xy 0 4.67)))
     (wall (+y p c) (+xy p l c)))))

(define (roof-framing p c l lvl h)
  (parameterize ((default-wall-family wall-framing)
                 (default-level-to-level-height h)
                 (current-level (level lvl))
                 (material/wall generic-metal))
    (union
     (wall p (+x p l))
     (wall (+x p l) (+xy p l c))
     (wall (+xy p l c) (+y p c))
     (wall (+y p c) p))))
  
;;INTERIOR
(define (interior-walls p c l lvl1 lvl2 h)
  (parameterize ((default-wall-family wall-type1)
                 (default-level-to-level-height h)
                 (current-level (level lvl1))
                 (material/wall dirty-white))
    (union
     (wall p (+y p c))
     (wall (+y p c) (+xy p l c))
     (wall (+xy p l c) (+x p l)
           (level lvl2) (level (+ lvl1 h)))
     (wall (+x p l) (+x p (- (/ l 2) 0.85))
           (level lvl2) (level (+ lvl1 h)))
     (wall (+x p (- (/ l 2) 0.85)) p))))

(define (interior-walls-allum p c l lvl h)
  (parameterize ((default-wall-family wall-type1)
                 (default-level-to-level-height h)
                 (current-level (level lvl))
                 (material/wall generic-metal))
    (union
     (wall p (+x p l))
     (wall (+x p l) (+xy p l c))
     (wall (+xy p l c) (+y p c)))))

(define (interior-walls-beam p c l lvl h)
  (parameterize ((default-wall-family wall-type2)
                 (default-level-to-level-height h)
                 (current-level (level lvl))
                 (material/wall dirty-white))
    (union
     (wall p (+y p c))
     (wall (+y p c) (+xy p l c))
     (wall (+xy p l c) (+x p l))
     (wall (+x p l) p))))


(define (interior-walls-windows p c l lvl h trans-material)
  (parameterize ((default-wall-family wall-type2)
                 (default-level-to-level-height h)
                 (current-level (level lvl))
                 (material/wall trans-material))
    (union
     (wall p (+x p (- l)))
     (wall p (+y p c)))))
     

;------------------------------------------------- CURTAIN WALLS ---------------------------------------------------;
(define (curtain-walls p c l h trans-material)
  (parameterize ((default-wall-family
                   (wall-family-element
                    (default-wall-family)
                    #:thickness 0.01))
                 (default-level-to-level-height h)
                 #;(current-level (level 0))
                 (material/wall trans-material))
    (union
     (wall (+x p (/ l 2)) (+x p l))
     (wall (+x p l) (+xy p l c)))))


;----------------------------------------------------- BEAMS -------------------------------------------------------;
;uma aproximação à viga existente no projecto terá a forma de um L
(define (beam-concrete-L p c l hM hm lM lm)
  (let ((seccaoL (surface-polygon (xy 0 0) (xy lM 0) (xy lM hM)
                                  (xy lm hM) (xy lm hm) (xy 0 hm)))
        (percurso (closed-line p (+y p c) (+xy p l c) (+x p l))))
    (add-radiance-shape!
     (shape-material (sweep percurso seccaoL)
                     concrete))))


;--------------------------------------------------- CEILLINGS -----------------------------------------------------;
;;FAMILIES
(define fam-ceiling-type1 (slab-family-element
                        (default-slab-family)
                        #:thickness 0.05))

(define fam-ceiling-type2 (slab-family-element
                        (default-slab-family)
                        #:thickness 0.25))

;;CEILLINGS
(define (ceiling-type2 p c l)
  (parameterize ((default-slab-family fam-ceiling-type2)
                 (material/slab-ceiling concrete)
                 (material/slab-floor concrete))
    (slab (list p (+y p c) (+xy p l c) (+x p l)))))

(define (ceiling-metallic-simple)
  (let
      ((p (xy 0.14 0.14))
       (c 37.2)
       (l 14.2))
      (box (+z p 8.36) l c 0.05)))

;WITH RECTANGULAR OPENING
(define (ceiling-metallic-rectangular p c l lvl
                                      p2 c2 l2
                                      trans-material)
  (add-radiance-shape!
   (shape-material
    (subtraction
     (box (+z p lvl) l c 0.05)
     (box p2 c2 l2 2))
    generic-metal))
  (add-radiance-shape!
   (shape-material
    (box p2 c2 l2 0.05)
   trans-material)))


;----------------------------------------------------- ROOFS -------------------------------------------------------;
(define (roof2 p length width height thickness)
  (let ((p1 (+yz p (/ width 2) height))
        (p2 (+y p width)))
      (thicken (extrusion (line p p1 p2) (vx length p))
               thickness)))

(define (roof-50)
  (roof2 (loc-from-o-vx-vy
          (xyz 0.165 22.325 9.23)
          (vy -1)
          (vx 1))
         22.16
         14.26
         0.42
         0.5))

(define (roof-20)
  (roof2 (loc-from-o-vx-vy
          (xyz 0.165 22.325 9.23)
          (vy -1)
          (vx 1))
         22.16
         14.26
         0.42
         0.2))

(define (roof-05)
  (roof2 (loc-from-o-vx-vy
          (xyz 0.165 22.325 9.38)
          (vy -1)
          (vx 1))
         22.16
         14.26
         0.42
         0.05))


;--------------------------------------------------- SKYLIGHTS -----------------------------------------------------;
;;RECTANGULAR
(define (roof-rectangular-skylight-glass p l c trans-material)
  (add-radiance-shape!
   (shape-material
    (intersection
     (roof-05)
     (box (+z p 2) l c -6))
    trans-material)))

(define (roof-rectangular-skylight-opening p l c trans-material)
  (add-radiance-shape!
   (shape-material
    (subtraction
     (roof-20)
     (box (+z p 2) l c -6))
    concrete))
  (roof-rectangular-skylight-glass p l c trans-material))

(define (roof-rectangular-skylight p height l c trans-material)
  (let ((e 0.05))
    (roof-rectangular-skylight-opening p l c trans-material)
    (add-radiance-shape!
     (shape-material
      (subtraction
       (subtraction
        (box (+xy p (- e) (- e)) (+ l (* 2 e)) (+ c (* 2 e)) (- height))
        (box p l c (- height)))
       (roof-50))
      dirty-white))))


;;CONIC
(define (union-map-division f t0 t1 n)
  (if (= n 0)
      (f t0)
      (if (= n -1)
          (empty-shape)
          (union (map-division f t0 t1 n)))))

(define (conic-skylight center height radius-m radius-M)
  (let ((e 0.05))
    (subtraction
     (cone-frustum center (+ radius-m e)
                   (+z center (- height)) (+ radius-M e))
     (cone-frustum center radius-m
                   (+z center (- height)) radius-M)
     (roof-50))))

(define (ceiling-metallic-conic p c l lvl
                                center height radius-m radius-M n-skylights d)
  (let ((e 0.05))
    (add-radiance-shape!
     (shape-material
      (subtraction
       (box (+z p lvl) l c 0.05)
       (union-map-division (lambda (y)
                             (cone-frustum (+y center y) (+ radius-m e)
                                           (+yz center y (- height)) (+ radius-M e)))
                           0 d (- n-skylights 1)))
      generic-metal))))

(define (roof-conic-skylight-glass center height radius-m radius-M n-skylights d trans-material)
  (let ((e 0.05))
    (add-radiance-shape!
     (shape-material
      (intersection
       (roof-05)
       (union-map-division (lambda (y)
                             (cone-frustum (+y center y) radius-m
                                           (+yz center y (- height)) radius-M))
                           0 d (- n-skylights 1)))
      trans-material))))

(define (roof-conic-skylight center height radius-m radius-M n-skylights d trans-material)
  (let ((e 0.05))
    (add-radiance-shape!
     (shape-material
      (subtraction
       (roof-20)
       (union-map-division (lambda (y)
                             (cone-frustum (+y center y) radius-m
                                           (+yz center y (- height)) radius-M))
                           0 d (- n-skylights 1)))
      concrete))
    (add-radiance-shape!
     (shape-material
      (union-map-division (lambda (y)
                            (conic-skylight (+y center y) height radius-m radius-M))
                          0 d (- n-skylights 1))
      dirty-white))
    (roof-conic-skylight-glass center height radius-m radius-M n-skylights d trans-material)))

  

;-------------------------------------------------------------------------------------------------------------------;
;                                                       FINAL                                                       ;                                                            
;-------------------------------------------------------------------------------------------------------------------;

;--------------------------------------------- BUILDING WITHOUT ROOF -----------------------------------------------;
(define (building-no-roof trans-material)
  ;;SLABS
  (slabs (xyz 0 0 0)
         37.53 14.59
         3
         22.525 7.5
         8.30)
  ;;PASSADIÇO
  (passadico (xyz 12.38 22.36 3.35)
             7.83
             1.5
             1.2
             0.1)
  ;;COLUMNS
  (columns-type1 (xyz 0 0 0)
                 5.37
                 37.53 14.59
                 6 3)
  (columns-type2 (xyz -0.355 0 0)
                 5.37
                 37.53
                 9)
  ;;WALLS
  (exterior-walls (xyz -0.175 -0.175 0)
                  37.88 14.765
                  5.37)
  (roof-framing (xy 0.315 22.475)
                7.6 13.96
                7.25 2.63)
  (interior-walls (xy 0.66 0.54)
                  36.23 13.37
                  3.25 5.37
                  5.13)
  (interior-walls-allum (xy 0.76 22.425)
                        7.70 1.80 
                        3.25 4)
  (interior-walls-beam (xy 0.14 0.14) 37.25 14.31
                       6.17 3.68)
  (muros (xyz 0 0 0)
         37.53 14.59
         5.55 3.45
         15.10
         2.5
         3.7)
  ;;WINDOWS/CURTAIN WALLS
  (curtain-walls (xyz 0 -0.005 0)
                 37.535 14.595
                 5.37
                 trans-material)
  ;;BEAMS
  (beam-concrete-L (xyz 0.4 0.4 5.37)
                   36.73 13.79
                   2.5 0.8
                   0.82 0.62)
  ;;CELLINGS
  (ceiling-type2 (xyz 0.465 22.625 7.50) 7.3 13.66)
  ;;ROOFS
  (add-radiance-shape!
   (shape-material
    (roof2 (loc-from-o-vx-vy
            (xyz 0.165 37.365 9.23)
            (vy -1)
            (vx 1))
           7.14
           14.26
           0.42
           0.2)
    concrete)))



;---------------------------------------------- RECTANGULAR SKYLIGHT -----------------------------------------------;
(define (building-rectangular-skylight skylight?
                                         p height l c
                                         interior?
                                         trans-material)
  (building-no-roof trans-material)
  (let ((e 0.05))
    (if skylight?
        (begin
          (roof-rectangular-skylight p height l c trans-material)
          (add-radiance-shape!
           (shape-material
            (subtraction
             (ceiling-metallic-simple)
             (box (+xy p (- e) (- e)) (+ l (* 2 e)) (+ c (* 2 e)) (- height)))
            generic-metal)))
        (begin
          (add-radiance-shape!
           (shape-material
            (roof-20)
            concrete))
          (add-radiance-shape!
           (shape-material
            (ceiling-metallic-simple)
            generic-metal))))
    (when interior?
      (interior-walls-windows (xy 13.5 1.09) 20 5.5 3.25 1.9 trans-material))))
     

;------------------------------------------------- CONIC SKYLIGHT --------------------------------------------------;
(define (building-conic-skylight n-skylights d
                                 center height radius-m radius-M
                                 interior?
                                 trans-material)
  (building-no-roof trans-material)
   (println (~a "n-skylights=" n-skylights "; d=" d "; center=" center "; height=" height "; radius-m=" radius-m "; radius-M=" radius-M "; material=" material-name))
      

  (let ((e 0.05))
    (ceiling-metallic-conic (xy 0.14 0.14) 37.2 14.2 8.36
                            center height radius-m radius-M n-skylights d)
    (roof-conic-skylight center height radius-m radius-M n-skylights d trans-material))

  (when interior?
    (interior-walls-windows (xy 13.5 1.09) 20 6 3.25 2 trans-material)))


;(building-conic-skylight 5 17.5 (xyz 2.75 2.49 9.65) 2.5 0.3 1 #f whitish-panel-25)


;------------------------------------------------- ANALYSIS --------------------------------------------------;
; 
;------------------------------------------------- -- --------------------------------------------------;
(define n-skylights (make-parameter 1))
(define d (make-parameter 17.5))
(define center-y (make-parameter 2.49))
(define height (make-parameter 1.5))
(define radius-M (make-parameter 0.3))
(define radius-m (make-parameter 0.2))
(define trans-material (make-parameter whitish-panel-25))

(define (udi-analysis)
  (parameterize
       ((current-location "C:\\Users\\catar\\Projects\\Research Projects\\MscThesis\\examples\\CaseStudyIP\\PRT_Lisboa.085360_INETI.epw")
        (current-occupancy "8to6withDST.60min.occ.csv")
        (material/ground grass)
        (analysis-nodes-height 1.5)
        (analyze-surfaces #f)
        (daysim-min-udi 0)
        (daysim-max-udi 220)
        (daysim-min-illuminance 200))
    (delete-all-shapes)
    (with-simulation
        (let-values (((df da cda udi)                      
                      (daysim-simulation 
                       (building-conic-skylight (n-skylights) (d) (xyz 2.49 (center-y) 9.65) (height) (radius-m) (radius-M) #f (trans-material)))))
          (* 100.0 udi)))))

(define run-udi-analysis
  (lambda ([nskylghts 5] [dist 17] [cy  2.49] [rm 0.5] [rM 0.9] [mat 0])
    (parameterize ([n-skylights nskylghts]
                   [d dist]
                   [center-y cy]
                   [radius-m rm]
                   [radius-M rM]
                   [trans-material (list-ref panels-list mat)])
      (println (~a "n-skylights=" (n-skylights) "; d=" (d) "; center-y=" (center-y) "; radius-m=" (radius-m) "; radius-M=" (radius-M) "; material=" (material-name (trans-material))))
      (udi-analysis))))


(define cmd-args (current-command-line-arguments))
(define nskylights (string->number (vector-ref cmd-args 0)))
(define dist (string->number (vector-ref cmd-args 1)))
(define cy (string->number (vector-ref cmd-args 2)))
(define rm (string->number (vector-ref cmd-args 3)))
(define rM (string->number (vector-ref cmd-args 4)))
(define mat (string->number (vector-ref cmd-args 5)))

(run-udi-analysis nskylights dist cy rm rM mat)

(run-udi-analysis)

;-------------------------------------------------------------------------------------------------------------------;
;                                                    SIMULATIONS                                                    ;                                                            
;-------------------------------------------------------------------------------------------------------------------;
(require racket/runtime-path)
(define-runtime-path radiance-results "PavPreto.3dm")

;NOTE: The sensors distance is changed in the Slabs function.


;-------------------------------------------------------------------------------------------------------------------;
;                                                 MATERIALS COST                                                    ;                                                            
;-------------------------------------------------------------------------------------------------------------------;
(define (rectangular-skylight-cost h l c)
  (+
   (* (* l c) 185)
   (* (* 2 h (+ l c)) 80)))

(define (conic-skylight-cost r R height n-skylights)
  (let ((s
         (sqrt (+
                (sqr height)
                (sqr (- R r))))))
    (*
     (+ (* (* pi (+ R r) s)
           80)
        (* (* pi (sqr r)) 185))
     n-skylights)))


;----------------------------------------------- OPTIMIZATION INFO -------------------------------------------------;
;   (building-rectangular-skylight skylight?
;                                  p height l c
;                                  interior?)
;   (rectangular-skylight-cost h l c)
;
;
;   skylight? #t
;
;   p (xyz 2.49 2.49 9.66)
;
;   HEIGHT
;   height default = 2.5
;   height min = 1.5
;   height max = 5
;
;   WIDHT
;   width min = 0.1
;   width max = 4
;    for aesthetical reasons it is best for width max be 1.5 or 2 
;
;   LENGHT
;   lenght min = 0.1
;   lenght max = 17.5
;   
;   interior? #f
;
;   (building-rectangular-skylight #t
;                                  (xyz 2.49 2.49 9.66) 2.5 l c
;                                  #f
;                                  whitish-panel-25)
;
;   (rectangular-skylight-cost 2.5 l c)
;
;
;   for w=3 * l=3
;     x min = 2.5
;     x max = 4
;     y min = 2.5
;     y max = 14                                                           
;-------------------------------------------------------------------------------------------------------------------;               