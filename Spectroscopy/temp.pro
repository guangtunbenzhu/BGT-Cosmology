; load and interpolate *ALL* quasar spectra
; output at both restframe and observer framer, with common central wavelength grid for each 
;-
pro jhusdss_hstfos_restobs_allqsospec, overwrite=overwrite, $
       doobserver=doobserver, dorest=dorest


;; qsopath
parent_path='~/SDATA/SDSS/AllInOne'
outfile_base = parent_path+'/AIO_QSO_HSTFOS'
wave_outfile = parent_path+'/AIO_CommonWave.fits'

channel = ['h130', 'h190', 'h270']
;; no header files, use explicit file names
 observer_outfile = outfile_base+'_ObserverFrame_Wave01100_03350A.fits'
rest_outfile_base = outfile_base+'_NEDzRestFrame.fits'

qsofile = '~/SDATA/Quasars/HSTFOS/table1_master.fits'
qso = mrdfits(qsofile, 1)
nqso = n_elements(qso)

if (file_test(wave_outfile)) then begin
   splog, "Warning: wavelength grid already exists. Use the existing one."
   wave = (mrdfits(wave_outfile, 1)).wave
   loglam = alog10(wave)
endif else begin
   loglam = jhusdss_get_loglam(minwave=448., maxwave=10404.)
   wave = 10.^loglam
   mwrfits, {wave:wave}, wave_outfile, /create
endelse
nwave = n_elements(loglam)

delta_z = 0.1
nzbin = fix((max(qso.z)+0.101)/delta_z)
print, "Memory used: ", memory(/current)/1024./1024., ' MB'

;; #######################
;; observer frame
if keyword_set(doobserver) then begin
;; #######################
print, "I am now working on the observer frame."

if (file_test(observer_outfile) and ~keyword_set(overwrite)) then begin
   splog, 'File exists. Use Overwrite to overwrite!'
   return
endif else begin
   splog, 'Will write into these files:'
   print, observer_outfile 
endelse

observer_wave_range = [1100., 3350.]
observer_loc = value_locate(wave, observer_wave_range)
observer_loglam = loglam[observer_loc[0]:observer_loc[1]+1]
observer_nwave = n_elements(observer_loglam)
observer_outstr = {ra:qso.ra2000, dec:qso.dec2000, z:qso.z, $
                   wave:wave[observer_loc[0]:observer_loc[1]+1], $
                   flux:fltarr(nqso, observer_nwave), $
                   ivar:fltarr(nqso, observer_nwave)}

observer_allflux_out = fltarr(nqso, observer_nwave)
observer_allivar_out = fltarr(nqso, observer_nwave)
                                                                                                                                                           24,4           0%
print, "Memory used: ", memory(/current)/1024./1024., ' MB'

;; observer frame
for i=0L, nqso-1L do begin
    tmp_allflux = fltarr(3, n_elements(observer_loglam))
    tmp_allivar = fltarr(3, n_elements(observer_loglam))
    for j=0L, 2L do begin
        tmpspec = jhusdss_hstfos_readspec(qso[i], channel[j], status=status)
        if (status eq 1) then begin
           tmpwave = tmpspec.wave
           tmpflux = tmpspec.flux
           tmpivar = (1.-(tmpspec.error eq 0.))/(tmpspec.error^2 + (tmpspec.error eq 0.))
           tmpmask = (tmpivar le 0.)
           curr_loglam = alog10(tmpwave)
           combine1fiber, curr_loglam, tmpflux, tmpivar, newloglam=observer_loglam, $
                          newflux=finterp, newivar=iinterp, maxiter=0, $
                          finalmask=tmpmask, andmask=maskinterp
           tmp_allflux[j,*] = finterp
           tmp_allivar[j,*] = iinterp
        endif
    endfor
    ;; Co-add
    tmp_outivar = total(tmp_allivar, 1)
    tmp_outflux = total(tmp_allflux*tmp_allivar, 1)/(tmp_outivar+(tmp_outivar eq 0.))*(tmp_outivar gt 0.)
    observer_allflux_out[i,*] = tmp_outflux*10.
    observer_allivar_out[i,*] = tmp_outivar/100.
endfor

observer_outstr.flux = observer_allflux_out
observer_outstr.ivar = observer_allivar_out
mwrfits, observer_outstr, observer_outfile, /create
delvar, observer_allflux_out
delvar, observer_allvar_out
delvar, observer_outstr

;; #######################
endif ;; observer frame
;; #######################

;; #######################
;; rest frame
if keyword_set(dorest) then begin
;; #######################
print, "I am now working on the rest frame."

wave_range_aa = [450., 900.]
rest_loc_aa = value_locate(wave, wave_range_aa)
rest_outfile_aa = repstr(rest_outfile_base, '.fits', '_Wave00450_00900A.fits')
wave_range_bb = [900., 1800.]
rest_loc_bb = value_locate(wave, wave_range_bb)
rest_outfile_bb = repstr(rest_outfile_base, '.fits', '_Wave00900_01800A.fits')
wave_range_cc = [1800., 3600.]
rest_loc_cc = value_locate(wave, wave_range_cc)
rest_outfile_cc = repstr(rest_outfile_base, '.fits', '_Wave01800_03600A.fits')
wave_range_dd = [3600., 7200.]
rest_loc_dd = value_locate(wave, wave_range_dd)
rest_outfile_dd = repstr(rest_outfile_base, '.fits', '_Wave03600_07200A.fits')

pro jhusdss_hstfos_visualselection

qsofile = '~/SDATA/Quasars/HSTFOS/table1_master.fits'
qso = mrdfits(qsofile, 1)
nqso = n_elements(qso)

observer_file = '~/SDATA/SDSS/AllInOne/AIO_QSO_HSTFOS_ObserverFrame_Wave01100_03350A.fits'
spec = mrdfits(observer_file, 1)

outfile = '~/SDATA/Quasars/HSTFOS/hstfos_master_visual.fits'
outstr = replicate({name:'Name', ra:0.D0, dec:0.D0, z:0., h130:0L, h190:0L, h270:0L, isgood:0L}, nqso)
outstr.name = qso.name
outstr.ra = qso.ra2000
outstr.dec = qso.dec2000
outstr.z = qso.z
outstr.h130 = qso.h130
outstr.h190 = qso.h190
outstr.h270 = qso.h270

isgood = 'n'
for i=0L, nqso-1L do begin
    djs_plot, spec.wave, smooth(spec.flux[i,*], 5), xra=[1150, 3300], xst=1, yst=1

    djs_oplot, [2796., 2796.], !y.crange, color='green', thick=2, linestyle=2
    djs_oplot, [2803., 2803.], !y.crange, color='green', thick=2, linestyle=2
    djs_oplot, [1260., 1260.], !y.crange, color='green', thick=2, linestyle=2
    djs_oplot, [1393., 1393.], !y.crange, color='green', thick=2, linestyle=2

    djs_oplot, [2796., 2796.]*(1.+qso[i].z), !y.crange, color='red'
    djs_oplot, [2803., 2803.]*(1.+qso[i].z), !y.crange, color='magenta'
    djs_oplot, [1550., 1550.]*(1.+qso[i].z), !y.crange, color='blue'
    djs_oplot, [1216., 1216.]*(1.+qso[i].z), !y.crange, color='green'
    djs_oplot, [1032., 1032.]*(1.+qso[i].z), !y.crange, color='yellow'

    ; print, 'absorption, (y/n)'
    ; read, absorption
    ; if (absorption eq 'y') then outstr[i].absorption = 1L
    print, 'isgood, (y/n):'
    read, isgood
    if (isgood eq 'y') then outstr[i].isgood= 1L
endfor

mwrfits, outstr, outfile, /create

end
