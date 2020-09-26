#Multilevel selection model incorporating Host control#

Original codes sourced from 'The role of multilevel selection in host microbiome evolution' by van Vliet and Doebeli (2019)

##Modifications##

<ul>
  <li>Addition of one host gene (Hg) per holobiont</li>
  <li>2 forms of the gene: 0 = no effect on the microbe, 1 = helpers face no cost of helping.</li>
  <li>Initial condition: Alternating 0,1 in holobiont population. Average Hg frequency = 0.5.</li>
  <li>Model run with same parameters as Fig 3 of the original paper.</li>
<?ul>

##Files##

<ul>
  <li> Results from changed model seen in Hgene01.png (helper frequency distribution) and Hgene01h.png (host gene distribution). </li>
  <li>Changes made to MLS_figure_3.py and MLS_static_fast.py.</li>
  <li>singleRunMLS.py to see details of each point simulation. Example figures HMselxx.png - 4 figures simulating each corner point of the modified fig 3. Green line in first graph of those figures represent the Hg frequency.</li>
</ul>

##Other modifications##

20 points per axis instead of 30 to reduce computation time.
